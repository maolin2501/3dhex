import json
import threading
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
import lattice_studio as app


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.DATA.mkdir(exist_ok=True)
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),app.Handler)
        threading.Thread(target=cls.server.serve_forever,daemon=True).start()
        cls.base=f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        app.stop_all()
        cls.server.shutdown()
        cls.server.server_close()

    def get(self, path):
        with urlopen(self.base+path) as r:
            return json.load(r)

    def post(self, path, data, token=app.TOKEN):
        req=Request(self.base+path,data=json.dumps(data).encode(),headers={'X-Lattice-Token':token,'Content-Type':'application/json'})
        with urlopen(req) as r:
            return json.load(r)

    def wait(self, job):
        for _ in range(200):
            status=self.get('/api/status?job='+job)
            if status['state']!='running':return status
            time.sleep(.1)
        self.fail('job timeout')

    def test_unauthorized_mutations_and_host(self):
        with self.assertRaises(HTTPError) as cm:
            self.post('/api/load',{'demo':True},token='wrong')
        self.assertEqual(cm.exception.code,403)
        with self.assertRaises(HTTPError) as cm:
            urlopen(Request(self.base+'/api/bootstrap',headers={'Host':'evil.example'}))
        self.assertEqual(cm.exception.code,403)
        with self.assertRaises(HTTPError):
            urlopen(self.base+'/vendor/../../lattice_studio.py')

    def test_bootstrap_exposes_updated_sample_limits(self):
        result = self.get('/api/bootstrap')
        self.assertEqual(result['sample_limits'], dict(preview=32_000_000, final=400_000_000))

    def test_import_generate_export_and_cancel(self):
        model=self.post('/api/load',{'demo':True})['job']
        loaded=self.wait(model)
        self.assertEqual(loaded['state'],'done',loaded)
        source=self.get(f'/api/body?model={model}&body=0')
        self.assertEqual(len(source['indices'])//3,len(source['face_ids']))
        cfg=dict(lattice='hex3d',cell=5,wall=1,axis='z',scope='faces',design_faces=[0],preserve_faces=[1],shell=1,pitch=.3,depth=6,quality='preview',
                 surface_cell=6,surface_width=.9,pore_rounding=.2,junction_blend=.2,edge_width=.8,memory_mode='chunked')
        payload=dict(model=model,body=0,config=cfg)
        est=self.post('/api/estimate',payload)
        self.assertTrue(est['allowed'])
        job=self.post('/api/generate',payload)['job']
        done=self.wait(job)
        self.assertEqual(done['state'],'done',done)
        self.assertEqual(done['result']['config']['unselected'],'perforate')
        self.assertEqual(done['result']['stats']['unselected'],'perforate')
        self.assertEqual(done['result']['config']['design_version'],12)
        self.assertEqual(done['result']['stats']['surface_cell'],6)
        self.assertEqual(done['result']['stats']['surface_width'],.9)
        self.assertEqual(done['result']['stats']['edge_width'],.8)
        self.assertEqual(done['result']['stats']['memory_mode'],'chunked')
        self.assertIn('performance',done['result'])
        self.assertFalse(list(app.job_path(job).glob('scratch-lowmem-*')))
        with urlopen(self.base+f'/artifacts/{job}/result.stl') as r:
            stl=r.read()
            self.assertEqual(len(stl),84+done['result']['stats']['faces']*50)
        with urlopen(self.base+f'/artifacts/{job}/display.glb') as r:
            self.assertEqual(r.read(4),b'glTF')
        repaired=self.post('/api/redisplay',dict(job=job))['job']
        self.assertNotEqual(repaired,job)
        checked=self.wait(repaired)
        self.assertEqual(checked['state'],'done',checked)
        self.assertEqual(checked['result']['redisplay_of'],job)
        self.assertEqual(checked['result']['display_quality']['version'],2)
        self.assertTrue(checked['result']['display_quality']['watertight'])
        with urlopen(self.base+f'/artifacts/{repaired}/result.stl') as r:
            self.assertEqual(r.read(),stl)
        with urlopen(self.base+f'/artifacts/{job}/result.stl') as r:
            self.assertEqual(r.read(),stl)
        job2=self.post('/api/generate',payload|{'config':cfg|{'pitch':.12,'quality':'final'}})['job']
        for _ in range(200):
            if list(app.job_path(job2).glob('scratch-lowmem-*')):break
            time.sleep(.02)
        self.assertTrue(list(app.job_path(job2).glob('scratch-lowmem-*')))
        stopped=self.post('/api/cancel',dict(job=job2))
        self.assertEqual(stopped.get('state'),'cancelled',stopped)
        self.assertIsNotNone(app.JOBS[job2]['process'].poll())
        self.assertFalse(list(app.job_path(job2).glob('scratch-lowmem-*')))
        with self.assertRaises(HTTPError):
            urlopen(self.base+f'/artifacts/{job2}/result.stl')
        rejected=self.post('/api/redisplay',dict(job=job2))
        self.assertIn('error',rejected)
        bad=self.post('/api/generate',dict(model=model,body=0,config=cfg|{'design_faces':[999]}))
        self.assertIn('error',bad)


if __name__=='__main__':unittest.main()
