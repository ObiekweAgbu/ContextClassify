# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['run.py'],
             pathex=['C:\\Context_Classify\\venv\\Lib\\tensorflow', 'C:\\Context_Classify'],
             binaries=[],
             datas=[],
             hiddenimports=['tensorflow', 'tensorflow.lite.python.lite'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='run',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
