# テスト専用：GUI をリハーサルモード（GPU 未使用）で 127.0.0.1:7862 に起動する
# 今の PC のコミット残量が 19GiB をわずかに下回るため、このプロセスの中だけ sys_mem の残量に +3GiB を足して、
# 確認パネル → 開始 → 進捗 → 完了 の画面の流れを確かめる。製品のコード（app.py・job_manager.py）は変更しない
import os, sys
os.environ.pop("AVATAR_ALLOW_GPU", None)
os.chdir(r"D:\Avatar-Pipeline\gui"); sys.path.insert(0, r"D:\Avatar-Pipeline\gui")
sys.argv = ["app.py", "--gpu-rehearsal"]
import job_manager as jm
_real = jm.sys_mem
def _test_mem():
    m = dict(_real()); m["headroom_gib"] += 3.0; m["_test_adjusted"] = True
    return m
jm.sys_mem = _test_mem
import app
assert jm.GPU_SESSION == {"enabled": True, "rehearsal": True}
app.demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7862, share=False, inbrowser=False,
                                                   allowed_paths=app.PREVIEW_ROOTS, show_api=False)
