{
  "targets": [{
    "target_name": "addon_win64",
    'include_dirs': [
      "<!@(node -p \"require('node-addon-api').include\")",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\include"
    ],
    'dependencies': ["<!(node -p \"require('node-addon-api').gyp\")"],
    'libraries': [
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\lib\\x64\\*.lib"
    ],
    'defines': [ 'NAPI_DISABLE_CPP_EXCEPTIONS' ],
    "sources": ["cuda.cc"]
  }]
}