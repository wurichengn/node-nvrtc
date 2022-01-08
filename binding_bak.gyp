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
  },{
    "target_name": "addon_linux_x86_64",
    'include_dirs': [
      "<!@(node -p \"require('node-addon-api').include\")",
      "/usr/local/cuda-10.2/targets/x86_64-linux/include"
    ],
    'dependencies': ["<!(node -p \"require('node-addon-api').gyp\")"],
    'libraries': [
      "/usr/local/cuda-10.2/targets/x86_64-linux/lib/*.so",
      "/usr/local/cuda-10.2/targets/x86_64-linux/lib/stubs/*.so"
    ],
    'defines': [ 'NAPI_DISABLE_CPP_EXCEPTIONS' ],
    "sources": ["cuda.cc"],
    'cflags_cc': [ '-frtti','-fexceptions', '-std=gnu++0x' ]
  }]
}