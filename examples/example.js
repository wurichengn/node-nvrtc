var NVRTC = require("../index.js");

/**可用的设备数量 */
var deviceCount = NVRTC.getDeviceCount();

if(deviceCount <= 0){
  throw new Error("无法运行例子，因为并没有找到合适的设备");
}

//查看所有设备的信息
for(var i = 0;i < deviceCount;i++){
  console.log(`设备${i}`,NVRTC.getDeviceProperties(i));
}


/**用于计算的数据 */
var data = new Float32Array([1,1,1,1,1,1,1,1]);
/**显存空间 */
var cudaData = new NVRTC.CudaBuffer(data.byteLength);
//将js数据写入到cuda显存
cudaData.writeData(data.buffer);

var texture = new NVRTC.CudaTexutre3D(cudaData,{x:2,y:2,z:2});



/**用于测试的cuda代码，计算数的平方 */
var code = `my_program
__global__
void my_kernel(float* data,cudaTextureObject_t tex) {
    data[threadIdx.x] = tex3D<char>(tex,1,1,1) + tex3D<char>(tex,1,1,1);
}`;

/**初始化计算用的启动器 */
var launcher = new NVRTC.CudaProgram(code).createKernel("my_kernel").createInstantiate([]).createLauncher([1,1,1],[data.length,1,1]);

//进行一次cuda运算
launcher.run(cudaData,texture);

//读取计算结果
cudaData.readData(data.buffer);
console.log(data);