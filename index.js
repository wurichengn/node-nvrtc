var os = require("os");
var osInfo = os.platform() + ":" + os.arch();
try{
    if(osInfo == "win32:x64"){
        var addon = require("./bin/addon_win64.node");
    }else if(osInfo == "linux:x64"){
        var addon = require("./bin/addon_linux_x86_64.node");
    }
}catch(e){
    throw new Error(`初始化cuda扩展失败，可能的原因：
1.没有安装支持cuda的显卡驱动
2.没有安装合适的vc环境
3.您的系统可能不是64位系统
4.还没有您系统对应的生产包，您需要配置cuda开发环境并且进入该项目修改相关配置重新编译`);
}

/**
 * 获取可用显卡数量
 * @type {()=>number}
 */
var getDeviceCount = addon.getDeviceCount;
module.exports.getDeviceCount = getDeviceCount;

/**
 * 获取当前正在使用的设备
 * @type {()=>number}
 */
var getDevice = addon.getDevice;
module.exports.getDevice = getDevice;

/**
 * 设置要使用的设备
 * @type {(device:number)=>void}
 */
var setDevice = addon.setDevice;
module.exports.setDevice = setDevice;

/**
 * 获取指定设备的信息
 * @type {(device:number)=>{}}
 */
var getDeviceProperties = addon.getDeviceProperties;
module.exports.getDeviceProperties = getDeviceProperties;

/**cuda程序 */
class CudaProgram{
    /**
     * 
     * @param {string} code cuda程序的代码
     * @param {(filename:string)=>(string|null)} fileCallback 引入文件回调函数，当有include文件时会通过这个回调函数处理
     */
    constructor(code,fileCallback){
        var self = this;
        /**Cuda程序句柄 */
        this.program = addon.createProgram(code,fileCallback);

        /**
         * 创建一个Cuda核心
         * @param {string} name 要创建核心的函数名
         * @returns {CudaKernel}
         */
        this.createKernel = function(name){
            return new CudaKernel(self,name);
        }
    }
}

module.exports.CudaProgram = CudaProgram;



/**cuda核心 */
class CudaKernel{
    /**
     * 
     * @param {CudaProgram} program 核心所属的程序
     * @param {string} name 核心的函数名
     */
    constructor(program,name){
        var self = this;
        /**核心所属的程序 */
        this.program = program;
        /**核心的函数名 */
        this.name = name;
        /**Cuda核心句柄 */
        this.kernel = addon.createKernel(program.program,name);
        
        /**
         * 创建一个运算实例
         * @param {[]} templates 要创建实例的模板参数
         * @returns {CudaInstantiate}
         */
        this.createInstantiate = function(templates){
            return new CudaInstantiate(self,templates);
        }
    }
}

module.exports.CudaKernel = CudaKernel;


/**Cuda 实例 */
class CudaInstantiate{
    /**
     * 
     * @param {CudaKernel} kernel 实例所属的核心
     * @param {[]} templates 实例的模板参数
     */
    constructor(kernel,templates){
        var self = this;
        /**实例所属的核心对象 */
        this.kernel = kernel;
        /**实例初始化时的模板参数 */
        this.templates = templates || [];
        var temps = [kernel.kernel,...templates];
        /**实例句柄 */
        this.instantiate = addon.createInstance.apply(addon,temps);

        /**
         * 创建启动器
         * @param {*} grid_size 启动器组尺寸
         * @param {*} block_size 块尺寸
         * @returns 
         */
        this.createLauncher = function(grid_size,block_size){
            return new CudaLauncher(self,grid_size,block_size);
        }
    }
}

module.exports.CudaInstantiate = CudaInstantiate;


/**Cuda启动器 */
class CudaLauncher{
    /**
     * 
     * @param {CudaInstantiate} instantiate 启动器所属的实例
     * @param {*} grid_size 启动器组的尺寸
     * @param {*} block_size 启动器块的尺寸
     */
    constructor(instantiate,grid_size,block_size){
        var self = this;
        /**启动器所属的实例 */
        this.instantiate = instantiate;
        /**启动器的分组尺寸 */
        this.grid_size = grid_size || [1,1,1];
        /**启动器区块的尺寸 */
        this.block_size = block_size || [1,1,1];
        /**启动器实例 */
        this.launcher = addon.createLauncher(instantiate.instantiate,grid_size,block_size);

        /**
         * 运行程序
         * @param  {...CudaBuffer} args 要运行的参数
         */
        this.run = function(...args){
            args = args.map(val=>val.buffer);
            addon.runLauncher(self.launcher,args);
        }
    }
}

module.exports.CudaLauncher = CudaLauncher;


/**Cuda缓冲区 */
class CudaBuffer{
    constructor(size){
        var self = this;
        /**缓冲区指针 */
        this.buffer = addon.createBuffer(size);
        /**缓冲区尺寸 */
        this.size = size;

        /**
         * 写入数据
         * @param {ArrayBuffer} buffer 要写入的buffer
         */
        this.writeData = function(buffer){
            //写入buffer
            addon.writeBuffer(self.buffer,buffer,buffer.byteLength);
        }

        /**
         * 读取数据
         * @param {ArrayBuffer} buffer 要存储读取的数据的buffer
         */
        this.readData = function(buffer){
            //读取buffer
            addon.readBuffer(self.buffer,buffer,buffer.byteLength);
        }
    }
}

module.exports.CudaBuffer = CudaBuffer;

