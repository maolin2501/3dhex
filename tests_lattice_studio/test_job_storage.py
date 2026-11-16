import json
import tempfile
import unittest
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch,Mock
import lattice_studio as app


class JobStorageTests(unittest.TestCase):
    def test_done_status_wins_exit_poll_race(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job='c'*24;directory=root/job;directory.mkdir()
            app.atomic_json(directory/'status.json',dict(state='running'))
            def poll():
                app.atomic_json(directory/'status.json',dict(state='done',result={'ok':True}))
                return 0
            worker=SimpleNamespace(poll=poll,returncode=0)
            with patch.object(app,'DATA',root),patch.object(app,'JOB_ROOT',None),patch.object(app,'JOBS',{job:{'process':worker}}):
                self.assertEqual(app.get_status(job)['state'],'done')
            self.assertEqual(json.loads((directory/'status.json').read_text())['state'],'done')

    def test_inherited_checkpoint_and_cycle_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a=root/('a'*24);b=root/('b'*24);a.mkdir();b.mkdir()
            (a/'pretrim').mkdir();app.atomic_json(a/'pretrim/ready.json',{})
            app.atomic_json(b/'request.json',dict(resume_from=str(a)))
            app.atomic_json(b/'status.json',dict(state='running'))
            worker=SimpleNamespace(poll=lambda:3221225477,returncode=3221225477)
            with patch.object(app,'DATA',root),patch.object(app,'JOB_ROOT',None),patch.object(app,'JOBS',{b.name:{'process':worker}}):
                self.assertEqual(app.checkpoint_source(b),a)
                failed=app.get_status(b.name)
                self.assertTrue(failed['checkpoint'])
                self.assertEqual(failed['checkpoint_job'],a.name)
                (a/'pretrim/ready.json').unlink()
                app.atomic_json(a/'request.json',dict(resume_from=str(b)))
                with self.assertRaisesRegex(ValueError,'循环'):app.checkpoint_source(b)

    def test_new_jobs_use_selected_disk_but_old_jobs_remain_readable(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);legacy=root/'old';external=root/'new'
            legacy.mkdir();external.mkdir()
            old_id='a'*24;(legacy/old_id).mkdir()
            worker=Mock();worker.poll.return_value=0
            with patch.object(app,'DATA',legacy),patch.object(app,'JOB_ROOT',external,create=True),patch.object(app,'JOBS',{}),patch('lattice_studio.subprocess.Popen',return_value=worker):
                new_id=app.start_job(dict(action='load',demo=True))['job']
                self.assertEqual(app.job_path(old_id),legacy/old_id)
                self.assertEqual(app.job_path(new_id),external/new_id)
                self.assertFalse((legacy/new_id).exists())
                self.assertEqual(json.loads((external/new_id/'request.json').read_text())['action'],'load')
                with self.assertRaises(ValueError):app.job_path('../bad')

    def test_cannot_change_disk_silently_while_server_running(self):
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)/'data';external=Path(folder)/'external'
            fake_web=Mock();fake_web.__truediv__=Mock(return_value=SimpleNamespace(is_file=lambda:True))
            with patch.object(app,'DATA',data),patch.object(app,'WEB',fake_web),patch.object(app,'JOB_ROOT',None),patch.object(sys,'argv',['lattice_studio.py','--no-browser','--job-root',str(external)]),patch.object(app,'LocalServer',side_effect=OSError('already running')):
                with self.assertRaisesRegex(SystemExit,'重启'):
                    app.main()
            self.assertFalse((data/'storage.json').exists())

    def test_fresh_ui_defaults_form_a_supported_combination(self):
        from html.parser import HTMLParser
        class Defaults(HTMLParser):
            def __init__(self):super().__init__();self.current=None;self.values={}
            def handle_starttag(self,tag,attrs):
                a=dict(attrs)
                if tag=='select':self.current=a.get('id')
                if tag=='option' and self.current:
                    if self.current not in self.values or 'selected' in a:self.values[self.current]=a.get('value')
            def handle_endtag(self,tag):
                if tag=='select':self.current=None
        parser=Defaults();parser.feed((app.WEB/'index.html').read_text(encoding='utf-8'))
        values=parser.values
        config=dict(lattice=values['latticeType'],seam_style=values['seamStyle'],axis='z',scope='body')
        self.assertEqual(app.validate_design(config)['seam_style'],'atlas')


if __name__=='__main__':unittest.main()
