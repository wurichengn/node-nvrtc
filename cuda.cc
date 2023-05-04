#include <napi.h>
#include "jitify.hpp"
#include "cuda_runtime.h"

// using namespace Napi;

/**��ǰ�ļ������Ӧ�������� */
Napi::Env * lastFileEnv;
/**��ǰ�ļ������Ӧ�Ļص����� */
Napi::Function lastFileCallback;
//�ļ���ȡ�ص�
std::istream* file_callback(std::string filename, std::iostream& tmp_stream) {
  //���û�лص�����������
  if(lastFileEnv == NULL) {return 0;}
  //���ûص�
  auto value = lastFileCallback.Call(lastFileEnv->Global(),{Napi::String::New(*lastFileEnv,filename)});
  //��������ַ��������
  if(!value.IsString()) {return 0;}
  //д������
  tmp_stream << value.As<Napi::String>().Utf8Value();
  return &tmp_stream;
}

//����cuda����
void NodeCudaError(Napi::Env env,cudaError_t error){
  if(error == cudaSuccess) {return;}
  std::string err(cudaGetErrorString(cudaGetLastError()));
  err += std::string(":") + std::string(__FILE__) + std::string(":") + std::to_string(__LINE__);
  Napi::TypeError::New(env,err).ThrowAsJavaScriptException();
}

//======��ȡ�豸����======
Napi::Value getDeviceProperties(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  //��ȡcuda�豸
  int device = args[0].As<Napi::Number>().Int32Value();
  cudaDeviceProp prop;
  NodeCudaError(env,cudaGetDeviceProperties(&prop,device));

  //���ؾ��
  Napi::Object re = Napi::Object::New(env);
  //д������
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

//======��ȡ�豸����======
Napi::Value getDeviceCount(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  //��ȡcuda�豸
  int count = 0;
  NodeCudaError(env,cudaGetDeviceCount(&count));
  //���ؾ��
  return Napi::Number::New(env,(size_t)count);
}

//======��ȡ��ǰʹ�õ��豸======
Napi::Value getDevice(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  //��ȡcuda�豸
  int device = 0;
  NodeCudaError(env,cudaGetDevice(&device));
  //���ؾ��
  return Napi::Number::New(env,(size_t)device);
}

//======���õ�ǰʹ�õ��豸======
void setDevice(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  //��ȡcuda�豸
  int device = args[0].As<Napi::Number>().Int32Value();
  NodeCudaError(env,cudaSetDevice(device));
}

//���õ�ǰ�Կ���ǰ���̵���������
void deviceReset(const Napi::CallbackInfo& args){
  cudaDeviceReset();
}

//��ȡ��ǰ�豸������
Napi::Value deviceGetLimit(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  //��ȡ����������
  int type = args[0].As<Napi::Number>().Int32Value();
  size_t value;
  NodeCudaError(env,cudaDeviceGetLimit(&value,(cudaLimit)type));
  return Napi::Number::New(env,value);
}

//���õ�ǰ�豸������
void deviceSetLimit(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  //��ȡ����������
  int type = args[0].As<Napi::Number>().Int32Value();
  //��ȡ����ֵ
  size_t value = args[1].As<Napi::Number>().Int32Value();
  NodeCudaError(env,cudaDeviceSetLimit((cudaLimit)type,value));
}

//======��������======
Napi::Value createProgram(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  //����Ĵ���
  auto str = args[0].As<Napi::String>().Utf8Value();

  //�ص�����
  if(args.Length() > 1 && args[1].IsFunction()){
    lastFileEnv = &env;
    lastFileCallback = args[1].As<Napi::Function>();
  }else{
    lastFileEnv = NULL;
  }
  
  jitify::experimental::Program *program = NULL;
  try{
    //����cuda����
    std::vector<std::string> opts;
    program = new jitify::experimental::Program(str, {}, opts,file_callback);
  }catch(std::runtime_error msg){
    Napi::TypeError::New(env,msg.what()).ThrowAsJavaScriptException();
  }

  //���ؾ��
  return Napi::Number::New(env,(size_t)program);
}





//======��������======
Napi::Value createKernel(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  //Ҫ�����ĺ��ķ�������
  auto str = args[1].As<Napi::String>().Utf8Value();

  //��������
  //jitify::experimental::Program * program = (jitify::experimental::Program *)args[0].As<Napi::Number>().Int64Value();
  //jitify::experimental::Kernel object = program->kernel(str);
  //jitify::experimental::Kernel * kernel = (jitify::experimental::Kernel *) malloc(sizeof(jitify::experimental::Kernel));
  //memcpy(kernel,&object,sizeof(jitify::experimental::Kernel));
  jitify::experimental::Program * program = NULL;
  jitify::experimental::Kernel * kernel = NULL;
  try{
    program = (jitify::experimental::Program *)args[0].As<Napi::Number>().Int64Value();
    kernel = new jitify::experimental::Kernel(program,str,std::vector<std::string>({}));
  }catch(std::runtime_error msg){
    Napi::TypeError::New(env,msg.what()).ThrowAsJavaScriptException();
  }

  return Napi::Number::New(env,(size_t)kernel);
}




//======����ʵ��======
Napi::Value createInstance(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  //��ʼ������ʵ���Ĳ���
  std::vector<std::string> instance_args;
  for(int i = 1;i < args.Length();i++){
    instance_args.push_back(args[i].As<Napi::String>().Utf8Value());
  }

  //����ʵ��
  jitify::experimental::Kernel * kernel = NULL;
  jitify::experimental::KernelInstantiation * instance = NULL;
  try{
    kernel = (jitify::experimental::Kernel *)args[0].As<Napi::Number>().Int64Value();
    instance = new jitify::experimental::KernelInstantiation(*kernel,instance_args);
  }catch(std::runtime_error msg){
    Napi::TypeError::New(env,msg.what()).ThrowAsJavaScriptException();
  }

  return Napi::Number::New(env,(size_t)instance);
}

//======��ȡʵ����Ϣ======
Napi::Value getInstancePTX(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  //Ҫ��ȡ��Ϣ��ʵ��
  jitify::experimental::KernelInstantiation * instance = (jitify::experimental::KernelInstantiation *)args[0].As<Napi::Number>().Int64Value();

  //��������
  Napi::Object re = Napi::Object::New(env);
  //д��PTX
  re.Set(Napi::String::New(env, "ptx"),Napi::String::New(env,instance->ptx()));
  //д�������ļ�
  Napi::Array v_files = Napi::Array::New(env);
  std::vector<std::string> link_files = instance->link_files();
  for(int i = 0;i < link_files.size();i++){
    v_files.Set(Napi::Number::New(env,i),Napi::String::New(env,link_files[i]));
  }
  re.Set(Napi::String::New(env, "link_files"),v_files);
  //д�������ļ�
  Napi::Array v_paths = Napi::Array::New(env);
  std::vector<std::string> link_paths = instance->link_paths();
  for(int i = 0;i < link_paths.size();i++){
    v_paths.Set(Napi::Number::New(env,i),Napi::String::New(env,link_paths[i]));
  }
  re.Set(Napi::String::New(env, "link_paths"),v_paths);
  return re;
}

//======ʵ�����л�======
Napi::Value serializeInstance(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  //Ҫ��ȡ��Ϣ��ʵ��
  jitify::experimental::KernelInstantiation * instance = (jitify::experimental::KernelInstantiation *)args[0].As<Napi::Number>().Int64Value();
  
  //���л�
  std::string code = instance->serialize();

  //ת��ΪarrayBuffer
  Napi::ArrayBuffer arrayBuffer = Napi::ArrayBuffer::New(env,code.size());
  void * bufferData = arrayBuffer.Data();
  memcpy(bufferData,code.c_str(),code.size());

  return arrayBuffer;
}

//======ʵ�������л�======
Napi::Value deserializeInstance(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  //ת��Ϊ�ַ���
  Napi::ArrayBuffer buffer = args[0].As<Napi::ArrayBuffer>();
  std::string code((char *)buffer.Data(),buffer.ByteLength());

  //�����л�
  try{
    jitify::experimental::KernelInstantiation object = jitify::experimental::KernelInstantiation::deserialize(code);
    jitify::experimental::KernelInstantiation * instance = (jitify::experimental::KernelInstantiation *)malloc(sizeof(jitify::experimental::KernelInstantiation));
    memcpy(instance,&object,sizeof(jitify::experimental::KernelInstantiation));
    return Napi::Number::New(env,(size_t)instance);
  }catch(std::runtime_error msg){
    Napi::TypeError::New(env,msg.what()).ThrowAsJavaScriptException();
  }

  return Napi::Number::New(env,0);
}


//======����������======
Napi::Value createLauncher(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  auto sg = args[1].As<Napi::Array>();
  auto bg = args[2].As<Napi::Array>();

  dim3 grid(sg.Get(0u).As<Napi::Number>().Uint32Value(),
            sg.Get(1u).As<Napi::Number>().Uint32Value(),
            sg.Get(2u).As<Napi::Number>().Uint32Value());
  dim3 block(bg.Get(0u).As<Napi::Number>().Uint32Value(),
            bg.Get(1u).As<Napi::Number>().Uint32Value(),
            bg.Get(2u).As<Napi::Number>().Uint32Value());

  //����ʵ��
  jitify::experimental::KernelInstantiation * instance = (jitify::experimental::KernelInstantiation *)args[0].As<Napi::Number>().Int64Value();
  jitify::experimental::KernelLauncher object = instance->configure(grid,block);
  jitify::experimental::KernelLauncher * launcher = (jitify::experimental::KernelLauncher *)malloc(sizeof(jitify::experimental::KernelLauncher));
  memcpy(launcher,&object,sizeof(jitify::experimental::KernelLauncher));

  return Napi::Number::New(env,(size_t)launcher);
}


//======���������ڴ�ռ�======
Napi::Value createBufferHost(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  void * buffer = NULL;
  NodeCudaError(env,cudaHostAlloc(&buffer,(size_t)args[0].As<Napi::Number>().Int64Value(),cudaHostAllocDefault ));

  return Napi::Number::New(env,(size_t)buffer);
}


//======�����ڴ�ռ�======
Napi::Value createBuffer(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  void * buffer = NULL;
  NodeCudaError(env,cudaMalloc(&buffer,(size_t)args[0].As<Napi::Number>().Int64Value()));

  return Napi::Number::New(env,(size_t)buffer);
}

//======д������======
void writeBuffer(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  void * buffer = (void **)args[0].As<Napi::Number>().Int64Value();
  size_t size = (size_t)args[2].As<Napi::Number>().Int64Value();
  void * data = args[1].As<Napi::ArrayBuffer>().Data();
  
  NodeCudaError(env,cudaMemcpy(buffer, data, size, cudaMemcpyHostToDevice));
}

//======�ͷ��ڴ�======
void freeBufferHost(const Napi::CallbackInfo& args){
  //��ȡenv
  
  Napi::Env env = args.Env();

  void * buffer = (void **)args[0].As<Napi::Number>().Int64Value();
  NodeCudaError(env,cudaFreeHost(buffer));
}

//======�ͷ��ڴ�======
void freeBuffer(const Napi::CallbackInfo& args){
  //��ȡenv
  
  Napi::Env env = args.Env();

  void * buffer = (void **)args[0].As<Napi::Number>().Int64Value();
  NodeCudaError(env,cudaFree(buffer));
}


//======��ȡ����======
void readBuffer(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  void * buffer = (void **)args[0].As<Napi::Number>().Int64Value();
  size_t size = (size_t)args[2].As<Napi::Number>().Int64Value();
  void * data = args[1].As<Napi::ArrayBuffer>().Data();
  NodeCudaError(env,cudaMemcpy(data,buffer, size, cudaMemcpyDeviceToHost));
}



//======��������======
Napi::Value createArray3D(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  size_t sizeX = (size_t)args[0].As<Napi::Number>().Int64Value();
  size_t sizeY = (size_t)args[1].As<Napi::Number>().Int64Value();
  size_t sizeZ = (size_t)args[2].As<Napi::Number>().Int64Value();

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<char>();
  cudaArray_t array = NULL;
  NodeCudaError(env,cudaMalloc3DArray(&array, &channelDesc, make_cudaExtent(sizeX*sizeof(char),sizeY,sizeZ), 0));

  return Napi::Number::New(env,(size_t)array);
}

//======д������======
void writeArray3D(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  cudaArray_t array = (cudaArray_t)args[0].As<Napi::Number>().Int64Value();
  void * data = args[1].As<Napi::ArrayBuffer>().Data();
  size_t sizeX = (size_t)args[2].As<Napi::Number>().Int64Value();
  size_t sizeY = (size_t)args[3].As<Napi::Number>().Int64Value();
  size_t sizeZ = (size_t)args[4].As<Napi::Number>().Int64Value();

  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr   = make_cudaPitchedPtr(data, sizeX*sizeof(char), sizeY, sizeZ);
  copyParams.dstArray = array;
  copyParams.extent   = make_cudaExtent(sizeX,sizeY,sizeZ);
  copyParams.kind     = cudaMemcpyHostToDevice;
  NodeCudaError(env,cudaMemcpy3D(&copyParams));
}



//======��ȡ����======
void readArray3D(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  cudaArray_t array = (cudaArray_t)args[0].As<Napi::Number>().Int64Value();
  void * data = args[1].As<Napi::ArrayBuffer>().Data();
  size_t sizeX = (size_t)args[2].As<Napi::Number>().Int64Value();
  size_t sizeY = (size_t)args[3].As<Napi::Number>().Int64Value();
  size_t sizeZ = (size_t)args[4].As<Napi::Number>().Int64Value();

  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcArray = array;
  copyParams.dstPtr   = make_cudaPitchedPtr(data, sizeX*sizeof(char), sizeY, sizeZ);
  copyParams.extent   = make_cudaExtent(sizeX,sizeY,sizeZ);
  copyParams.kind     = cudaMemcpyDeviceToHost;
  NodeCudaError(env,cudaMemcpy3D(&copyParams));
}


//������άbuffer
Napi::Value createBuffer3D(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  cudaPitchedPtr * ptr = new cudaPitchedPtr();
  cudaExtent size;
  size.width = (size_t)args[0].As<Napi::Number>().Int64Value();
  size.height = (size_t)args[1].As<Napi::Number>().Int64Value();
  size.depth = (size_t)args[2].As<Napi::Number>().Int64Value();
  NodeCudaError(env,cudaMalloc3D(ptr,size));

  Napi::Object re = Napi::Object::New(env);
  re.Set(Napi::String::New(env, "index"),Napi::Number::New(env,(size_t)ptr));
  re.Set(Napi::String::New(env, "ptr"),Napi::Number::New(env,(size_t)ptr->ptr));
  re.Set(Napi::String::New(env, "pitch"),Napi::Number::New(env,(size_t)ptr->pitch));
  re.Set(Napi::String::New(env, "xsize"),Napi::Number::New(env,(size_t)ptr->xsize));
  re.Set(Napi::String::New(env, "ysize"),Napi::Number::New(env,(size_t)ptr->ysize));
  return re;
}


//д����άbuffer
void writeBuffer3D(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  cudaPitchedPtr * ptr = (cudaPitchedPtr *)args[0].As<Napi::Number>().Int64Value();
  void * data = args[1].As<Napi::ArrayBuffer>().Data();

  cudaExtent size;
  size.width = (size_t)args[2].As<Napi::Number>().Int64Value();
  size.height = (size_t)args[3].As<Napi::Number>().Int64Value();
  size.depth = (size_t)args[4].As<Napi::Number>().Int64Value();
  size_t width = (size_t)args[5].As<Napi::Number>().Int64Value();

  cudaMemcpy3DParms parms = {0};
  parms.srcPtr = make_cudaPitchedPtr(data, size.width, width, size.height);
  parms.dstPtr = *ptr;
  parms.extent = size;
  parms.kind = cudaMemcpyHostToDevice;
  NodeCudaError(env,cudaMemcpy3D(&parms));
}


//��ȡ��άbuffer
void readBuffer3D(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  cudaPitchedPtr * ptr = (cudaPitchedPtr *)args[0].As<Napi::Number>().Int64Value();
  void * data = args[1].As<Napi::ArrayBuffer>().Data();

  cudaExtent size;
  size.width = (size_t)args[2].As<Napi::Number>().Int64Value();
  size.height = (size_t)args[3].As<Napi::Number>().Int64Value();
  size.depth = (size_t)args[4].As<Napi::Number>().Int64Value();
  size_t width = (size_t)args[5].As<Napi::Number>().Int64Value();

  cudaMemcpy3DParms parms = {0};
  parms.dstPtr = make_cudaPitchedPtr(data, size.width, width, size.height);
  parms.srcPtr = *ptr;
  parms.extent = size;
  parms.kind = cudaMemcpyDeviceToHost;
  NodeCudaError(env,cudaMemcpy3D(&parms));
}


Napi::Value createTexture3D(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  cudaPitchedPtr * ptr = (cudaPitchedPtr *)args[0].As<Napi::Number>().Int64Value();
  size_t length = (size_t)args[1].As<Napi::Number>().Int64Value();

  cudaTextureObject_t texture;
  
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<char>();

  cudaResourceDesc    texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));
  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = ptr;
  texRes.res.linear.sizeInBytes = length;
  texRes.res.linear.desc = desc;

  cudaTextureDesc     texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));
  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModePoint;
  texDescr.addressMode[0] = cudaAddressModeMirror;
  texDescr.addressMode[1] = cudaAddressModeMirror;
  texDescr.addressMode[2] = cudaAddressModeMirror;
  texDescr.readMode = cudaReadModeElementType;

  NodeCudaError(env,cudaCreateTextureObject(&texture, &texRes, &texDescr, NULL));
  return Napi::Number::New(env,(size_t)texture);
}


//======���к���======
Napi::Value runKernel(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  //Ҫ���صĶ���
  Napi::Object re = Napi::Object::New(env);
  re.Set(Napi::String::New(env,"code"),Napi::Number::New(env,0));

  //�����ַ
  jitify::experimental::Program * program = (jitify::experimental::Program *)args[0].As<Napi::Number>().Int64Value();
  //��������
  auto kname = args[1].As<Napi::String>().Utf8Value();

  //��ʼ������ʵ���ķ��Ͳ���
  auto templates = args[2].As<Napi::Array>();
  std::vector<std::string> instance_template;
  for(int i = 0;i < templates.Length();i++){
    std::cout << templates.Get(i).As<Napi::String>().Utf8Value() << std::endl;
    instance_template.push_back(templates.Get(i).As<Napi::String>().Utf8Value());
  }

  //��ʼ��ʵ������
  auto arguments = args[3].As<Napi::Array>();
  std::vector<void *> instance_args;
  void * addrs[256];
  for(int i = 0;i < arguments.Length();i++){
    addrs[i] = (void *)arguments.Get(i).As<Napi::Number>().Int64Value();
    instance_args.push_back((void *)&addrs[i]);
  }

  //������㷶Χ
  dim3 grid(1);
  dim3 block(1);

  //����
  auto res = program->kernel(kname).instantiate(instance_template).configure(grid, block).launch(instance_args);
  if(res != CUDA_SUCCESS){
    const char* str;
    cuGetErrorName(res, &str);
    re.Set(Napi::String::New(env,"err"),Napi::String::New(env,str));
    re.Set(Napi::String::New(env,"code"),Napi::Number::New(env,-1));
  }
  return re;
}



//======����ʵ��======
Napi::Value runInstance(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  //Ҫ���صĶ���
  Napi::Object re = Napi::Object::New(env);
  re.Set(Napi::String::New(env,"code"),Napi::Number::New(env,0));

  //�����ַ
  jitify::experimental::KernelInstantiation * instance = (jitify::experimental::KernelInstantiation *)args[0].As<Napi::Number>().Int64Value();

  //��ʼ��ʵ������
  auto arguments = args[1].As<Napi::Array>();
  std::vector<void *> instance_args;
  void * addrs[256];
  for(int i = 0;i < arguments.Length();i++){
    addrs[i] = (void *)arguments.Get(i).As<Napi::Number>().Int64Value();
    instance_args.push_back((void *)&addrs[i]);
  }

  //������㷶Χ
  dim3 grid(1);
  dim3 block(1);

  //����
  auto res = instance->configure(grid, block).launch(instance_args);
  if(res != CUDA_SUCCESS){
    const char* str;
    cuGetErrorName(res, &str);
    re.Set(Napi::String::New(env,"err"),Napi::String::New(env,str));
    re.Set(Napi::String::New(env,"code"),Napi::Number::New(env,-1));
  }
  return re;
}



//======����������======
Napi::Value runLauncher(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  //Ҫ���صĶ���
  Napi::Object re = Napi::Object::New(env);
  re.Set(Napi::String::New(env,"code"),Napi::Number::New(env,0));

  //�����ַ
  jitify::experimental::KernelLauncher * launcher = (jitify::experimental::KernelLauncher *)args[0].As<Napi::Number>().Int64Value();

  //��ʼ��ʵ������
  auto arguments = args[1].As<Napi::Array>();
  std::vector<void *> instance_args;
  void * addrs[256];
  for(int i = 0;i < arguments.Length();i++){
    addrs[i] = (void *)arguments.Get(i).As<Napi::Number>().Int64Value();
    instance_args.push_back((void *)&addrs[i]);
  }

  //������㷶Χ
  dim3 grid(1);
  dim3 block(1);

  //����
  auto res = launcher->launch(instance_args);
  if(res != CUDA_SUCCESS){
    const char* str;
    cuGetErrorName(res, &str);
    re.Set(Napi::String::New(env,"err"),Napi::String::New(env,str));
    re.Set(Napi::String::New(env,"code"),Napi::Number::New(env,-1));
    re.Set(Napi::String::New(env,"file"),Napi::String::New(env,__FILE__));
    re.Set(Napi::String::New(env,"line"),Napi::Number::New(env,__LINE__));
  }
  return re;
}




Napi::Value test(const Napi::CallbackInfo& args){
  //��ȡenv
  Napi::Env env = args.Env();

  //Ҫ���صĶ���
  Napi::Object re = Napi::Object::New(env);
  re.Set(Napi::String::New(env,"code"),Napi::Number::New(env,0));

  void * d_data = (void *)args[2].As<Napi::Number>().Int64Value();

  //��ʼ������ʵ���Ĳ���
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

//CUDA����
Napi::Object CudaTest(const Napi::CallbackInfo& args){
  Napi::Env env = args.Env();
  //Ҫ���صĶ���
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

//�����ʼ������
Napi::Object Init(Napi::Env env, Napi::Object exports) {
  //��ʼ��cuda
  // void * buffer;
  // cudaMalloc(&buffer,1);

  exports.Set(Napi::String::New(env, "CudaTest"),Napi::Function::New(env, CudaTest));
  exports.Set(Napi::String::New(env, "createProgram"),Napi::Function::New(env, createProgram));
  exports.Set(Napi::String::New(env, "createKernel"),Napi::Function::New(env, createKernel));
  exports.Set(Napi::String::New(env, "createInstance"),Napi::Function::New(env, createInstance));
  exports.Set(Napi::String::New(env, "createLauncher"),Napi::Function::New(env, createLauncher));

  exports.Set(Napi::String::New(env, "getInstancePTX"),Napi::Function::New(env, getInstancePTX));
  exports.Set(Napi::String::New(env, "serializeInstance"),Napi::Function::New(env, serializeInstance));
  exports.Set(Napi::String::New(env, "deserializeInstance"),Napi::Function::New(env, deserializeInstance));

  exports.Set(Napi::String::New(env, "createBuffer"),Napi::Function::New(env, createBuffer));
  exports.Set(Napi::String::New(env, "createBufferHost"),Napi::Function::New(env, createBufferHost));
  exports.Set(Napi::String::New(env, "writeBuffer"),Napi::Function::New(env, writeBuffer));
  exports.Set(Napi::String::New(env, "readBuffer"),Napi::Function::New(env, readBuffer));
  exports.Set(Napi::String::New(env, "freeBuffer"),Napi::Function::New(env, freeBuffer));
  exports.Set(Napi::String::New(env, "freeBufferHost"),Napi::Function::New(env, freeBufferHost));

  exports.Set(Napi::String::New(env, "createBuffer3D"),Napi::Function::New(env, createBuffer3D));
  exports.Set(Napi::String::New(env, "writeBuffer3D"),Napi::Function::New(env, writeBuffer3D));
  exports.Set(Napi::String::New(env, "readBuffer3D"),Napi::Function::New(env, readBuffer3D));

  exports.Set(Napi::String::New(env, "createArray3D"),Napi::Function::New(env, createArray3D));
  exports.Set(Napi::String::New(env, "writeArray3D"),Napi::Function::New(env, writeArray3D));
  exports.Set(Napi::String::New(env, "readArray3D"),Napi::Function::New(env, readArray3D));

  exports.Set(Napi::String::New(env, "createTexture3D"),Napi::Function::New(env, createTexture3D));

  exports.Set(Napi::String::New(env, "runKernel"),Napi::Function::New(env, runKernel));
  exports.Set(Napi::String::New(env, "runLauncher"),Napi::Function::New(env, runLauncher));
  exports.Set(Napi::String::New(env, "runInstance"),Napi::Function::New(env, runInstance));
  exports.Set(Napi::String::New(env, "test"),Napi::Function::New(env, test));
  exports.Set(Napi::String::New(env, "getDeviceCount"),Napi::Function::New(env, getDeviceCount));
  exports.Set(Napi::String::New(env, "getDevice"),Napi::Function::New(env, getDevice));
  exports.Set(Napi::String::New(env, "setDevice"),Napi::Function::New(env, setDevice));
  exports.Set(Napi::String::New(env, "deviceGetLimit"),Napi::Function::New(env, deviceGetLimit));
  exports.Set(Napi::String::New(env, "deviceSetLimit"),Napi::Function::New(env, deviceSetLimit));
  exports.Set(Napi::String::New(env, "deviceReset"),Napi::Function::New(env, deviceReset));
  exports.Set(Napi::String::New(env, "getDeviceProperties"),Napi::Function::New(env, getDeviceProperties));

  return exports;
}

//��Init��Ϊ��ʼ������
NODE_API_MODULE(addon, Init)