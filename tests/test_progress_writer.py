import json
import os

import pytest

from scripts import research_detection as module


def test_progress_writer_retries_temporary_reader_lock(tmp_path, monkeypatch):
    path=tmp_path/'progress.json'; path.write_text('old')
    replace=module.os.replace; attempts=[]; sleeps=[]
    def locked(source,destination):
        attempts.append(str(source))
        if len(attempts)<3: raise PermissionError('temporary reader lock')
        return replace(source,destination)
    monkeypatch.setattr(module.os,'replace',locked)
    monkeypatch.setattr(module.time,'sleep',sleeps.append)
    module.write_json(path,{'completed':3})
    assert json.loads(path.read_text())=={'completed':3}
    assert sleeps==[.02,.05] and len(attempts)==3
    assert list(tmp_path.iterdir())==[path]


def test_progress_writer_preserves_previous_state_when_lock_persists(tmp_path, monkeypatch):
    path=tmp_path/'progress.json'; path.write_text('previous')
    calls=[]
    def locked(*args):
        calls.append(1); raise PermissionError('persistent lock')
    monkeypatch.setattr(module.os,'replace',locked)
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError): module.write_json(path,{'completed':4})
    assert len(calls)==7 and path.read_text()=='previous'
    assert list(tmp_path.iterdir())==[path]


def test_progress_writer_does_not_retry_unrelated_io_errors(tmp_path, monkeypatch):
    calls=[]
    def fail(*args):
        calls.append(1); raise OSError('unrelated failure')
    monkeypatch.setattr(module.os,'replace',fail)
    with pytest.raises(OSError,match='unrelated'): module.write_json(tmp_path/'p.json',{})
    assert calls==[1] and list(tmp_path.iterdir())==[]


@pytest.mark.skipif(os.name!='nt',reason='Windows file sharing contract')
def test_progress_writer_recovers_from_real_windows_reader_lock(tmp_path):
    import ctypes
    from ctypes import wintypes
    import threading
    path=tmp_path/'progress.json'; path.write_text('previous')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    # A real reader that permits read sharing but not deletion/replacement.
    handle=kernel.CreateFileW(str(path),0x80000000,1,None,3,0x80,None)
    assert handle not in (None,ctypes.c_void_p(-1).value)
    closed=[]
    timer=threading.Timer(.08,lambda:closed.append(bool(kernel.CloseHandle(handle))))
    timer.start()
    try:
        module.write_json(path,{'completed':5})
    finally:
        timer.join()
    assert closed==[True] and json.loads(path.read_text())=={'completed':5}
    assert list(tmp_path.iterdir())==[path]
