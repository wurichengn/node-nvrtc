#include <napi.h>
#include "jitify.hpp"
#include "cuda_runtime.h"

using namespace Napi;

/**当前文件处理对应的上下文 */
Napi::Env * lastFileEnv;
/**当前文件处理对应的回调函数 */
Napi::Function lastFileCallback;
//文件读取回调
std::istream* file_callback(std::string filename, std::iostream& tmp_stream) {
  //如果没有回调处理则跳过
  if(lastFileEnv == NULL) {return 0;}
  //调用回调
  auto value = lastFileCallback.Call(lastFileEnv->Global(),{Napi::String::New(*lastFileEnv,filename)});
  //如果不是字符串则结束
  if(!value.IsString()) {return 0;}
  //写入内容
  tmp_stream << value.As<Napi::String>().Utf8Value();
  return &tmp_stream;
}

//处理cuda错误
void NodeCudaError(Napi::Env env,cudaError_t error){
  if(error == cudaSuccess) {return;}
  Napi::TypeError::New(env,cudaGetErrorString(cudaGetLastError())).ThrowAsJavaScriptException();
}

//======获取设备数量======
Napi::Value getDeviceProperties(const Napi::CallbackInfo& args){
  //获取env
  Napi::Env env = args.Env();

  //获取cuda设备
  int device = args[0].As<Napi::Number>().Int32Value();
  cudaDeviceProp prop;
  NodeCudaError(env,cudaGetDeviceProperties(&prop,device));

  //返回句柄
  Napi::Object re = Napi::Object::New(env);
  //写入属性
  re.Set(Napi::String::New(env, "name"),Napi::String::New(env,prop.name));
  re.Set(Napi::String::New(env, "totalGlobalMem"),Napi::Number::New(env,prop.totalGlobalMem));
  re.Set(Napi::String::New(env, "sharedMemPerBlock"),Napi::Number::New(env,prop.sharedMemPerBlock));
  re.Set(Napi::String::New(env, "regsPerBlock"),Napi::Number::New(env,prop.regsPerBlock));
  re.Set(Napi::String::New(env, "warpSize"),Napi::Number::New(env,prop.warpSize));
  re.Set(Napi::String::New(env, "memPitch"),Napi::Number::New(env,prop.memPitch));
  re.Set(Napi::String::New(env, "pciBusID"),Napi::Number::New(env,prop.pciBusID));
  re.Set(Napi::String::New(env, "pciDeviceID"),Napi::Number::New(env,prop.pciDeviceID));
  re.Set(Napi::String::New(env, "pciDomainID"),Napi::Number::New(env,prop.pciDomainID));

  return re;
}

//======获取设备数量======
Napi::Value getDeviceCount(const Napi::CallbackInfo& args){
  //获取env
  Napi::Env env = args.Env();

  //获取cuda设备
  int count = 0;
  NodeCudaError(env,cudaGetDeviceCount(&count));
  //返回句柄
  return Napi::Number::New(env,(size_t)count);
}

//======获取当前使用的设备======
Napi::Value getDevice(const Napi::CallbackInfo& args){
  //获取env
  Napi::Env env = args.Env();

  //获取cuda设备
  int device = 0;
  NodeCudaError(env,cudaGetDevice(&device));
  //返回句柄
  return Napi::Number::New(env,(size_t)device);
}

//======设置当前使用的设备======
void setDevice(const Napi::CallbackInfo& args){
  //获取env
  Napi::Env env = args.Env();

  //获取cuda设备
  int device = args[0].As<Napi::Number>().Int32Value();
  NodeCudaError(env,cudaSetDevice(device));
}

//======创建程序======
Napi::Value createProgram(const Napi::CallbackInfo& args){
  //获取env
  Napi::Env env = args.Env();

  //程序的代码
  auto str = args[0].As<Napi::String>().Utf8Value();

  //回调处理
  if(args.Length() > 1 && args[1].IsFunction()){
    lastFileEnv = &env;
    lastFileCallback = args[1].As<Napi::Function>();
  }else{
    lastFileEnv = NULL;
  }
  
  //创建cuda程序
  std::vector<std::string> opts;
  jitify::experimental::Program *program = new jitify::experimental::Program(str, {}, opts,file_callback);

  //返回句柄
  return Napi::Number::New(env,(size_t)program);
}





//======创建核心======
Napi::Value createKernel(const Napi::CallbackInfo& args){
  //获取env
  Napi::Env env = args.Env();

  //要创建的核心方法名称
  auto str = args[1].As<Napi::String>().Utf8Value();

  //创建核心
  //jitify::experimental::Program * program = (jitify::experimental::Program *)args[0].As<Napi::Number>().Int64Value();
  //jitify::experimental::Kernel object = program->kernel(str);
  //jitify::experimental::Kernel * kernel = (jitify::experimental::Kernel *) malloc(sizeof(jitify::experimental::Kernel));
  //memcpy(kernel,&object,sizeof(jitify::experimental::Kernel));
  jitify::experimental::Program * program = (jitify::experimental::Program *)args[0].As<Napi::Number>().Int64Value();
  auto kernel = new jitify::experimental::Kernel(program,str,std::vector<std::string>({}));

  return Napi::Number::New(env,(size_t)kernel);
}




//======创建实例======
Napi::Value createInstance(const Napi::CallbackInfo& args){
  //获取env
  Napi::Env env = args.Env();

  //初始化核心实例的参数
  std::vector<std::string> instance_args;
  for(int i = 1;i < args.Length();i++){
    instance_args.push_back(args[i].As<Napi::String>().Utf8Value());
  }

  //创建实例
  jitify::experimental::Kernel * kernel = (jitify::experimental::Kernel *)args[0].As<Napi::Number>().Int64Value();
  jitify::experimental::KernelInstantiation * instance = new jitify::experimental::KernelInstantiation(*kernel,instance_args);

  return Napi::Number::New(env,(size_t)instance);
}




//======创建启动器======
Napi::Value createLauncher(const Napi::CallbackInfo& args){
  //获取env
  Napi::Env env = args.Env();

  auto sg = args[1].As<Napi::Array>();
  auto bg = args[2].As<Napi::Array>();

  dim3 grid(sg.Get(0u).As<Napi::Number>().Uint32Value(),
            sg.Get(1u).As<Napi::Number>().Uint32Value(),
            sg.Get(2u).As<Napi::Number>().Uint32Value());
  dim3 block(bg.Get(0u).As<Napi::Number>().Uint32Value(),
            bg.Get(1u).As<Napi::Number>().Uint32Value(),
            bg.Get(2u).As<Napi::Number>().Uint32Value());

  //创建实例
  jitify::experimental::KernelInstantiation * instance = (jitify::experimental::KernelInstantiation *)args[0].As<Napi::Number>().Int64Value();
  jitify::experimental::KernelLauncher object = instance->configure(grid,block);
  jitify::experimental::KernelLauncher * launcher = (jitify::experimental::KernelLauncher *)malloc(sizeof(jitify::experimental::KernelLauncher));
  memcpy(launcher,&object,sizeof(jitify::experimental::KernelLauncher));

  return Napi::Number::New(env,(size_t)launcher);
}



//======申请内存空间======
Napi::Value createBuffer(const Napi::CallbackInfo& args){
  //获取env
  Napi::Env env = args.Env();

  void * buffer;
  NodeCudaError(env,cudaMalloc(&buffer,(size_t)args[0].As<Napi::Number>().Int64Value()));

  return Napi::Number::New(env,(size_t)buffer);
}

//======写入数据======
void writeBuffer(const Napi::CallbackInfo& args){
  //获取env
  Napi::Env env = args.Env();

  void * buffer = (void **)args[0].As<Napi::Number>().Int64Value();
  size_t size = (size_t)args[2].As<Napi::Number>().Int64Value();
  void * data = args[1].As<Napi::ArrayBuffer>().Data();
  NodeCudaError(env,cudaMemcpy(buffer, data, size, cudaMemcpyHostToDevice));
}

//======释放内存======
void freeBuffer(const Napi::CallbackInfo& args){
  //获取env
  
  Napi::Env env = args.Env();

  void * buffer = (void **)args[0].As<Napi::Number>().Int64Value();
  NodeCudaError(env,cudaFree(buffer));
}


//======读取数据======
void readBuffer(const Napi::CallbackInfo& args){
  //获取env
  Napi::Env env = args.Env();

  void * buffer = (void **)args[0].As<Napi::Number>().Int64Value();
  size_t size = (size_t)args[2].As<Napi::Number>().Int64Value();
  void * data = args[1].As<Napi::ArrayBuffer>().Data();
  NodeCudaError(env,cudaMemcpy(data,buffer, size, cudaMemcpyDeviceToHost));
}


//======运行核心======
Napi::Value runKernel(const Napi::CallbackInfo& args){
  //获取env
  Napi::Env env = args.Env();

  //要返回的对象
  Napi::Object re = Napi::Object::New(env);
  re.Set(Napi::String::New(env,"code"),Napi::Number::New(env,0));

  //程序地址
  jitify::experimental::Program * program = (jitify::experimental::Program *)args[0].As<Napi::Number>().Int64Value();
  //核心名称
  auto kname = args[1].As<Napi::String>().Utf8Value();

  //初始化核心实例的泛型参数
  auto templates = args[2].As<Napi::Array>();
  std::vector<std::string> instance_template;
  for(int i = 0;i < templates.Length();i++){
    std::cout << templates.Get(i).As<Napi::String>().Utf8Value() << std::endl;
    instance_template.push_back(templates.Get(i).As<Napi::String>().Utf8Value());
  }

  //初始化实例参数
  auto arguments = args[3].As<Napi::Array>();
  std::vector<void *> instance_args;
  void * addrs[256];
  for(int i = 0;i < arguments.Length();i++){
    addrs[i] = (void *)arguments.Get(i).As<Napi::Number>().Int64Value();
    instance_args.push_back((void *)&addrs[i]);
  }

  //构造计算范围
  dim3 grid(1);
  dim3 block(1);

  //运行
  auto res = program->kernel(kname).instantiate(instance_template).configure(grid, block).launch(instance_args);
  if(res != CUDA_SUCCESS){
    const char* str;
    cuGetErrorName(res, &str);
    re.Set(Napi::String::New(env,"err"),Napi::String::New(env,str));
    re.Set(Napi::String::New(env,"code"),Napi::Number::New(env,-1));
  }
  return re;
}



//======运行实例======
Napi::Value runInstance(const Napi::CallbackInfo& args){
  //获取env
  Napi::Env env = args.Env();

  //要返回的对象
  Napi::Object re = Napi::Object::New(env);
  re.Set(Napi::String::New(env,"code"),Napi::Number::New(env,0));

  //程序地址
  jitify::experimental::KernelInstantiation * instance = (jitify::experimental::KernelInstantiation *)args[0].As<Napi::Number>().Int64Value();

  //初始化实例参数
  auto arguments = args[1].As<Napi::Array>();
  std::vector<void *> instance_args;
  void * addrs[256];
  for(int i = 0;i < arguments.Length();i++){
    addrs[i] = (void *)arguments.Get(i).As<Napi::Number>().Int64Value();
    instance_args.push_back((void *)&addrs[i]);
  }

  //构造计算范围
  dim3 grid(1);
  dim3 block(1);

  //运行
  auto res = instance->configure(grid, block).launch(instance_args);
  if(res != CUDA_SUCCESS){
    const char* str;
    cuGetErrorName(res, &str);
    re.Set(Napi::String::New(env,"err"),Napi::String::New(env,str));
    re.Set(Napi::String::New(env,"code"),Napi::Number::New(env,-1));
  }
  return re;
}



//======运行启动器======
Napi::Value runLauncher(const Napi::CallbackInfo& args){
  //获取env
  Napi::Env env = args.Env();

  //要返回的对象
  Napi::Object re = Napi::Object::New(env);
  re.Set(Napi::String::New(env,"code"),Napi::Number::New(env,0));

  //程序地址
  jitify::experimental::KernelLauncher * launcher = (jitify::experimental::KernelLauncher *)args[0].As<Napi::Number>().Int64Value();

  //初始化实例参数
  auto arguments = args[1].As<Napi::Array>();
  std::vector<void *> instance_args;
  void * addrs[256];
  for(int i = 0;i < arguments.Length();i++){
    addrs[i] = (void *)arguments.Get(i).As<Napi::Number>().Int64Value();
    instance_args.push_back((void *)&addrs[i]);
  }

  //构造计算范围
  dim3 grid(1);
  dim3 block(1);

  //运行
  auto res = launcher->launch(instance_args);
  if(res != CUDA_SUCCESS){
    const char* str;
    cuGetErrorName(res, &str);
    re.Set(Napi::String::New(env,"err"),Napi::String::New(env,str));
    re.Set(Napi::String::New(env,"code"),Napi::Number::New(env,-1));
  }
  return re;
}




Napi::Value test(const Napi::CallbackInfo& args){
  //获取env
  Napi::Env env = args.Env();

  //要返回的对象
  Napi::Object re = Napi::Object::New(env);
  re.Set(Napi::String::New(env,"code"),Napi::Number::New(env,0));

  void * d_data = (void *)args[2].As<Napi::Number>().Int64Value();

  //初始化核心实例的参数
  std::vector<std::string> instance_args;
  for(int i = 3;i < args.Length();i++){
    instance_args.push_back(args[i].As<Napi::String>().Utf8Value());
  }
  jitify::experimental::Program * program = (jitify::experimental::Program *)args[0].As<Napi::Number>().Int64Value();
  auto kname = args[1].As<Napi::String>().Utf8Value();

  dim3 grid(1);
  dim3 block(1);
  auto res = program->kernel(kname).instantiate(instance_args).configure(grid, block).launch(d_data);
  if(res != CUDA_SUCCESS){
    const char* str;
    cuGetErrorName(res, &str);
    re.Set(Napi::String::New(env,"err"),Napi::String::New(env,str));
    re.Set(Napi::String::New(env,"code"),Napi::Number::New(env,-1));
  }
  return re;
}

//CUDA测试
Napi::Object CudaTest(const Napi::CallbackInfo& args){
  Napi::Env env = args.Env();
  //要返回的对象
  Napi::Object re = Napi::Object::New(env);
  auto str = args[0].As<Napi::String>().Utf8Value();
  
  std::vector<std::string> opts;
  jitify::experimental::Program *program = new jitify::experimental::Program(str, {}, opts);
  //auto program = jitify::experimental::Program::deserialize(program_orig.serialize());
  float h_data = 5;
  float* d_data;
  cudaMalloc((void**)&d_data, sizeof(float));
  cudaMemcpy(d_data, &h_data, sizeof(float), cudaMemcpyHostToDevice);
  dim3 grid(1);
  dim3 block(1);
  std::cout << jitify::reflection::reflect(jitify::reflection::type_of(*d_data)) << std::endl;
  auto kernel_inst_orig = program->kernel("my_kernel").instantiate(3, jitify::reflection::type_of(*d_data));
  auto kernel_inst = jitify::experimental::KernelInstantiation::deserialize(kernel_inst_orig.serialize());
  auto res = kernel_inst.configure(grid, block).launch(d_data);
  if(res != CUDA_SUCCESS){
    const char* str;
    cuGetErrorName(res, &str);
    re.Set(Napi::String::New(env,"err"),Napi::String::New(env,str));
  }else{
    cudaMemcpy(&h_data, d_data, sizeof(float), cudaMemcpyDeviceToHost);
    re.Set(Napi::String::New(env,"data"),Napi::Number::New(env,h_data));
  }
  cudaFree(d_data);
  return re;
}

//定义初始化方法
Napi::Object Init(Napi::Env env, Napi::Object exports) {
  //初始化cuda
  void * buffer;
  cudaMalloc(&buffer,1);

  exports.Set(Napi::String::New(env, "CudaTest"),Napi::Function::New(env, CudaTest));
  exports.Set(Napi::String::New(env, "createProgram"),Napi::Function::New(env, createProgram));
  exports.Set(Napi::String::New(env, "createKernel"),Napi::Function::New(env, createKernel));
  exports.Set(Napi::String::New(env, "createInstance"),Napi::Function::New(env, createInstance));
  exports.Set(Napi::String::New(env, "createLauncher"),Napi::Function::New(env, createLauncher));
  exports.Set(Napi::String::New(env, "createBuffer"),Napi::Function::New(env, createBuffer));
  exports.Set(Napi::String::New(env, "writeBuffer"),Napi::Function::New(env, writeBuffer));
  exports.Set(Napi::String::New(env, "readBuffer"),Napi::Function::New(env, readBuffer));
  exports.Set(Napi::String::New(env, "freeBuffer"),Napi::Function::New(env, freeBuffer));
  exports.Set(Napi::String::New(env, "runKernel"),Napi::Function::New(env, runKernel));
  exports.Set(Napi::String::New(env, "runLauncher"),Napi::Function::New(env, runLauncher));
  exports.Set(Napi::String::New(env, "runInstance"),Napi::Function::New(env, runInstance));
  exports.Set(Napi::String::New(env, "test"),Napi::Function::New(env, test));
  exports.Set(Napi::String::New(env, "getDeviceCount"),Napi::Function::New(env, getDeviceCount));
  exports.Set(Napi::String::New(env, "getDevice"),Napi::Function::New(env, getDevice));
  exports.Set(Napi::String::New(env, "setDevice"),Napi::Function::New(env, setDevice));
  exports.Set(Napi::String::New(env, "getDeviceProperties"),Napi::Function::New(env, getDeviceProperties));

  return exports;
}

//以Init作为初始化方法
NODE_API_MODULE(addon, Init)