var NVRTC = require("../index.js");


var voxel = new Uint8Array([1,2,3,4,5,6,7,8]);
var buffer = new NVRTC.CudaBuffer3D({x:2,y:2,z:2},1);
buffer.writeData(voxel.buffer);
var texture = new NVRTC.CudaBufferTexture3D(buffer);


/**用于测试的cuda代码，计算数的平方 */
var code = `my_program
__global__
void my_kernel(cudaPitchedPtr * pptr,cudaTextureObject_t texture) {
  char * ptr = (char *)pptr->ptr;
  ptr[-5] = 100;
}`;

/**初始化计算用的启动器 */
var launcher = new NVRTC.CudaProgram(code).createKernel("my_kernel").createInstantiate([]).createLauncher([1,1,1],[1,1,1]);

//进行一次cuda运算
launcher.run(buffer,texture);

//读取计算结果
buffer.readData(voxel.buffer);
console.log(voxel);