# node-nvrtc 让Node.js使用NVRTC的扩展

`node-nvrtc`是一个简易的使用`nvrtc`的`node.js`扩展，目前只包含简单的`nvrtc`功能和`cuda`内存交换相关功能。



## 安装

```javascript
npm install node-nvrtc
```

目前已经编译过的环境有`win-x86_64`以及`linux-x86_64`，其他环境需要手动根据cuda开发环境重新进行编译。

 - 需要注意的是`nvrtc`本身并不支持32位系统

## 使用

基本的使用方式如下

```javascript
var NVRTC = require("node-nvrtc");

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
var data = new Float32Array([1,2,3,4,5]);
/**显存空间 */
var cudaData = new NVRTC.CudaBuffer(data.byteLength);
//将js数据写入到cuda显存
cudaData.writeData(data.buffer);



/**用于测试的cuda代码，计算数的平方 */
var code = `my_program
__global__
void my_kernel(float* data) {
    data[threadIdx.x] = data[threadIdx.x] * data[threadIdx.x];
}`;

/**初始化计算用的启动器 */
var launcher = new NVRTC.CudaProgram(code).createKernel("my_kernel").createInstantiate([]).createLauncher([1,1,1],[data.length,1,1]);

//进行一次cuda运算
launcher.run(cudaData);

//读取计算结果
cudaData.readData(data.buffer);
console.log(data);
```