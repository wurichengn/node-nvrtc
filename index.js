var os = require("os");
var child_process = require("child_process");
var process = require("process");
var osInfo = os.platform() + ":" + os.arch();
try{
    if(osInfo == "win32:x64"){
        process.env.Path = __dirname + "\\lib;" + process.env.Path;
        var addon = require("./bin/addon_win64.node");
    }else if(osInfo == "linux:x64"){
        var addon = require("./bin/addon_linux_x86_64.node");
    }else{
        throw new Error("不支持的系统!");
    }
}catch(e){
    throw new Error(`初始化cuda扩展失败，可能的原因：
1.没有安装对应版本的cuda（默认为10.02）
2.没有安装合适的vc环境
3.您的系统可能不是64位系统
4.还没有您系统对应的生产包或者和您的cuda版本不符，您需要配置cuda开发环境并且进入该项目修改相关配置重新编译`);
}

/**
 * 重置当前选中设备当前进程的所有内容
 * @type {()=>number}
 */
 var deviceReset = addon.deviceReset;
 module.exports.deviceReset = deviceReset;

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
 * 获取设备的限制
 * @type {(type:number)=>void}
 */
 var deviceGetLimit = addon.deviceGetLimit;
 module.exports.deviceGetLimit = deviceGetLimit;

/**
 * 设置设备的限制
 * @type {(type:number,value:number)=>void}
 */
var deviceSetLimit = addon.deviceSetLimit;
module.exports.deviceSetLimit = deviceSetLimit;

/**设备限制枚举映射 */
module.exports.cudaLimit = {
    /** 线程堆栈大小 */
    cudaLimitStackSize:0,
    /** 线程打印内容尺寸 */
    cudaLimitPrintfFifoSize:1,
    /** 线程内部动态申请内存大小 */
    cudaLimitMallocHeapSize:2,
    /** 设备运行时同步深度 */
    cudaLimitDevRuntimeSyncDepth:3,
    /** 设备运行时挂起的启动计数 */
    cudaLimitDevRuntimePendingLaunchCount:4,
    /** 一个介于 0 和 128 之间的值，指示 L2 的最大获取粒度（以字节为单位） */
    cudaLimitMaxL2FetchGranularity:5,
    /** L2 持久行缓存大小的字节大小 */
    cudaLimitPersistingL2CacheSize:6
};

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
        this.templates = this.templates.map(v => (v + ""));
        var temps = [kernel.kernel,...this.templates];
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
            var re = addon.runLauncher(self.launcher,args);
            if(re.code != 0){
                throw new Error(re.err);
            }
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

        /**
         * 释放显存
         */
        this.destory = function(){
            addon.freeBuffer(this.buffer);
        }
    }
}

module.exports.CudaBuffer = CudaBuffer;


/**Cuda三维缓冲区 */
class CudaBuffer3D{
    /**
     * @param {{x:number,y:number,z:number}} size 数组尺寸
     * @param {number} unitSize 每个单位元素的字节数
     */
    constructor(size,unitSize = 1){
        var self = this;
        /**缓冲区指针 */
        this.instance = addon.createBuffer3D(size.x * unitSize,size.y,size.z);
        
        var ptrBuffer = new CudaBuffer(4 * 8);
        ptrBuffer.writeData(new BigUint64Array([
            BigInt(this.instance.ptr),
            BigInt(this.instance.pitch),
            BigInt(this.instance.xsize),
            BigInt(this.instance.ysize),
            BigInt(size.z)
        ]).buffer);

        this.buffer = ptrBuffer.buffer;
        /**缓冲区尺寸 */
        this.size = size;
        /**单位元素字节数 */
        this.unitSize = unitSize;

        /**
         * 写入数据
         * @param {ArrayBuffer} buffer 要写入的buffer
         */
        this.writeData = function(buffer){
            //写入buffer
            addon.writeBuffer3D(self.instance.index,buffer,size.x * unitSize,size.y,size.z,size.x);
        }

        /**
         * 读取数据
         * @param {ArrayBuffer} buffer 要存储读取的数据的buffer
         */
        this.readData = function(buffer){
            //读取buffer
            addon.readBuffer3D(self.instance.index,buffer,size.x * unitSize,size.y,size.z,size.x);
        }
    }
}

module.exports.CudaBuffer3D = CudaBuffer3D;


/**Cuda三维贴图 */
class CudaBufferTexture3D{
    /**
     * @param {CudaBuffer3D} cudaBuffer
     */
    constructor(cudaBuffer){
        var self = this;
        /**对应的cudaBuffer */
        this.cudaBuffer = cudaBuffer;
        /**显存占用的字节数 */
        this.length = cudaBuffer.instance.pitch * cudaBuffer.size.y * cudaBuffer.size.z;
        /**贴图指针 */
        this.buffer = addon.createTexture3D(this.cudaBuffer.buffer,this.length);
    }
}

module.exports.CudaBufferTexture3D = CudaBufferTexture3D;


/**Cuda三维贴图 */
class CudaTexutre3D{
    /**
     * @param {CudaBuffer} cudaBuffer 
     * @param {{x:number,y:number,z:number}} size
     */
    constructor(cudaBuffer,size){
        var self = this;
        /**对应的cudaBuffer */
        this.cudaBuffer = cudaBuffer;
        /**贴图尺寸 */
        this.size = size;
        /**贴图指针 */
        this.buffer = addon.createTexture3D(this.cudaBuffer.buffer,this.cudaBuffer.size,size.x,size.y,size.z);
    }
}

module.exports.CudaTexutre3D = CudaTexutre3D;


/**Cuda三维数组 */
class CudaArray3D{
    /**
     * @param {{x:number,y:number,z:number}} size 数组的尺寸
     */
    constructor(size){
        var self = this;
        /**数组指针 */
        this.buffer = addon.createArray3D(size.x,size.y,size.z);
        /**缓冲区尺寸 */
        this.size = size;

        /**
         * 写入数据
         * @param {ArrayBuffer} buffer 要写入的buffer
         */
        this.writeData = function(buffer){
            //写入buffer
            addon.writeArray3D(self.buffer,buffer,size.x,size.y,size.z);
        }

        /**
         * 读取数据
         * @param {ArrayBuffer} buffer 要存储读取的数据的buffer
         */
        this.readData = function(buffer){
            //读取buffer
            addon.readArray3D(self.buffer,buffer,size.x,size.y,size.z);
        }
    }
}

module.exports.CudaArray3D = CudaArray3D;