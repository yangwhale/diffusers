"""
Wan 视频生成模型的 JAX/TorchAX 实现脚本
这个脚本使用 JAX 后端来运行 Wan 文本到视频生成模型，支持多设备分布式推理
"""

import re
import math
import torch
import torchax  # PyTorch 到 JAX 的桥接库
import time
import jax

from jax.sharding import NamedSharding, PartitionSpec as P

from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from jax.tree_util import register_pytree_node
from transformers import modeling_outputs

# 定义分片轴的名称，用于多设备并行计算
axis = 'axis'

# Transformer 模型的权重分片配置
# 这个字典定义了如何在多个设备上分布 Transformer 的不同层的权重
# 格式：'层名称的正则表达式': (分片维度配置)
# axis 表示沿着该轴分片，None 表示不分片，() 表示完全复制
transformer_shardings = {
    # 注释掉的部分是完全复制的权重（为了速度考虑）
    # 'scale_shift_table': (), # 缩放偏移表，形状: (1, 2, 1536)
    # 'patch_embedding.weight': (), # 补丁嵌入权重，形状: (1536, 16, 1, 2, 2)
    # 'patch_embedding.bias': (), # 补丁嵌入偏置，形状: (1536)
    
    # 条件嵌入器 - 时间嵌入器的权重分片
    r'condition_embedder.time_embedder.linear_1.weight': (axis, None), # 第一个线性层权重，形状: (1536, 256)
    r'condition_embedder.time_embedder.linear_1.bias': (axis,), # 第一个线性层偏置，形状: (1536)
    r'condition_embedder.time_embedder.linear_2.weight': (None, axis), # 第二个线性层权重，形状: (1536, 1536)
    # 'condition_embedder.time_embedder.linear_2.bias': (), # 第二个线性层偏置（复制）
    
    # 条件嵌入器 - 文本嵌入器的权重分片
    r'condition_embedder.text_embedder.linear_1.weight': (axis, None), # 文本嵌入第一层权重，形状: (1536, 4096)
    r'condition_embedder.text_embedder.linear_1.bias': (axis, ), # 文本嵌入第一层偏置，形状: (1536)
    r'condition_embedder.text_embedder.linear_2.weight': (None, axis), # 文本嵌入第二层权重，形状: (1536, 1536)
    # 'condition_embedder.text_embedder.linear_2.bias': (), # 文本嵌入第二层偏置（复制）
    
    # Transformer 块的自注意力层权重分片
    # \d+ 匹配任意数字，表示所有 Transformer 块
    r'blocks.\d+.attn1.to_q.weight': (axis, None), # 查询投影权重，形状: (1536, 1536)
    r'blocks.\d+.attn1.to_q.bias': (axis, ), # 查询投影偏置，形状: (1536)
    r'blocks.\d+.attn1.to_k.weight': (axis, ), # 键投影权重，形状: (1536, 1536)
    r'blocks.\d+.attn1.to_k.bias': (axis, ), # 键投影偏置，形状: (1536)
    r'blocks.\d+.attn1.to_v.weight': (axis, ), # 值投影权重，形状: (1536, 1536)
    r'blocks.\d+.attn1.to_v.bias': (axis, ), # 值投影偏置，形状: (1536)
    # to_out 有两个子模块：第一个是线性层，第二个是 dropout
    r'blocks.\d+.attn1.to_out.0.weight': (None, axis), # 输出投影权重，形状: (1536, 1536)
    
    # 交叉注意力层权重分片（用于处理文本条件）
    r'blocks.\d+.attn2.to_q.weight': (axis, ), # 交叉注意力查询权重
    r'blocks.\d+.attn2.to_q.bias': (axis, ), # 交叉注意力查询偏置
    r'blocks.\d+.attn2.to_k.weight': (axis, ), # 交叉注意力键权重
    r'blocks.\d+.attn2.to_k.bias': (axis, ), # 交叉注意力键偏置
    r'blocks.\d+.attn2.to_v.weight': (axis, ), # 交叉注意力值权重
    r'blocks.\d+.attn2.to_v.bias': (axis, ), # 交叉注意力值偏置
    r'blocks.\d+.attn2.to_out.0.weight': (None, axis), # 交叉注意力输出权重
    
    # 前馈网络（FFN）权重分片
    r'blocks.\d+.ffn.net.0.proj.weight': (axis,), # FFN 投影层权重，形状: (8960, 1536)
    r'blocks.\d+.ffn.net.0.proj.bias': (axis, ), # FFN 投影层偏置，形状: (8960)
    r'blocks.\d+.ffn.net.2.weight': (None, axis), # FFN 第二层权重，形状: (1536, 8960)
    
    # 输出投影层（通常复制）
    # 'proj_out.weight': (), # 输出投影权重，形状: (64, 1536)
    # 'proj_out.bias': (), # 输出投影偏置，形状: (64)
}

# 文本编码器的权重分片配置
# 这里配置 T5 文本编码器的权重如何在设备间分布
text_encoder_shardings = {
    'shared.weight': (axis, ), # 共享嵌入权重，形状: (256384, 4096)
    
    # 编码器块的自注意力权重
    'encoder.block.*.layer.*.SelfAttention.q.weight': (axis, ), # 查询权重，形状: (4096, 4096)
    'encoder.block.*.layer.*.SelfAttention.k.weight': (axis, ), # 键权重，形状: (4096, 4096)
    'encoder.block.*.layer.*.SelfAttention.v.weight': (axis, ), # 值权重，形状: (4096, 4096)
    'encoder.block.*.layer.*.SelfAttention.o.weight': (None, axis), # 输出权重，形状: (4096, 4096)
    
    # 前馈网络权重
    'encoder.block.*.layer.*.DenseReluDense.wi_0.weight': (axis, ), # 第一个输入权重，形状: (10240, 4096)
    'encoder.block.*.layer.*.DenseReluDense.wi_1.weight': (axis, ), # 第二个输入权重，形状: (10240, 4096)
    'encoder.block.*.layer.*.DenseReluDense.wo.weight': (None, axis), # 输出权重，形状: (4096, 10240)
}


def _shard_weight_dict(weight_dict, sharding_dict, mesh):
    """
    根据分片配置将权重字典分布到多个设备上
    
    Args:
        weight_dict: 包含模型权重的字典
        sharding_dict: 分片配置字典，定义每个权重如何分片
        mesh: JAX 设备网格
    
    Returns:
        分片后的权重字典
    """
    result = {}
    for k, v in weight_dict.items():
        # 遍历分片配置，找到匹配的权重名称
        for target, sharding in sharding_dict.items():
            if re.fullmatch(target, k) is not None:
                # 应用指定的分片策略
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                break
        else:
            # 如果没有找到匹配的分片配置，则完全复制到所有设备
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))

        result[k] = v
    return result


def flatten_model_output(obj):
    """
    将模型输出对象扁平化为 JAX pytree 格式
    这是为了让 transformers 的输出对象能够与 JAX 兼容
    """
    return obj.to_tuple(), type(obj)

def unflatten_model_output(aux, children):
    """
    从扁平化的 pytree 重构模型输出对象
    """
    return aux(*children)

# 注册 transformers 模型输出为 JAX pytree 节点
# 这样 JAX 就能正确处理 transformers 的输出对象
register_pytree_node(
    modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
    flatten_model_output,
    unflatten_model_output)

def make_key(name):
    """
    将具体的层索引替换为通配符，用于生成分片配置的键
    例如：'blocks.0.attn1.weight' -> 'blocks.*.attn1.weight'
    """
    return re.sub('\.\d+\.', '.*.', name)

  
def _get_weights_of_linear(module):
    """
    递归获取模块中所有线性层的权重
    
    Args:
        module: PyTorch 模块
    
    Returns:
        包含所有线性层权重的字典
    """
    result = {}

    def fn(start_path, module):
        if isinstance(module, torch.nn.Linear):
            # 如果是线性层，获取其参数
            for k, v in module.named_parameters():
                start_path.append(k)
                key = '.'.join(start_path)
                result[key] = v
                start_path.pop()
        else:
            # 递归处理子模块
            for name, child in module.named_children():
                start_path.append(name)
                fn(start_path, child)
                start_path.pop()
    fn([], module)
    return result


def _print_weights(module):
    """
    打印模块中所有权重的形状和数据类型
    用于调试和生成分片配置
    """
    all_buffers = dict(module.named_parameters())
    all_buffers.update(module.named_buffers())
    result = {}
    for k, v in all_buffers.items():
        result[make_key(k)] = (v.shape, v.dtype)
    print('{')
    for k, v in result.items():
        print(f"'{k}': (), # {v}")
    print('}')

def main():
    """主函数：设置模型、编译并运行视频生成"""
    
    # 设置默认数据类型为 bfloat16 以节省内存
    torch.set_default_dtype(torch.bfloat16)
    
    # 可用模型：Wan-AI/Wan2.1-T2V-14B-Diffusers（14B参数）, Wan-AI/Wan2.1-T2V-1.3B-Diffusers（1.3B参数）
    model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    
    # 加载 VAE（变分自编码器）用于图像编码/解码
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    
    # 流偏移参数：5.0 用于 720P，3.0 用于 480P
    flow_shift = 5.0
    
    # 设置调度器，用于控制去噪过程
    scheduler = UniPCMultistepScheduler(
        prediction_type='flow_prediction', 
        use_flow_sigmas=True, 
        num_train_timesteps=1000, 
        flow_shift=flow_shift
    )
    
    # 加载完整的 Wan 管道
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = scheduler

    # 以下代码用于调试：打印模型权重信息（已注释）
    # print('vae=====')
    # _print_weights(pipe.vae)
    # print('trans===')
    # print(_get_weights_of_linear(pipe.transformer).keys())
    # print('encoder===')
    # _print_weights(pipe.text_encoder)
    # return

    def _move_module(module):
        """
        将 PyTorch 模块转换为 JAX 格式
        这个函数将模型权重从 PyTorch 张量转换为 JAX 数组
        """
        with jax.default_device('cpu'):
            state_dict = module.state_dict()
            state_dict = env.to_xla(state_dict)  # 转换为 XLA 格式
            module.load_state_dict(state_dict, assign=True)

    # 启用 TorchAX 全局模式，允许 PyTorch 代码在 JAX 后端运行
    torchax.enable_globally()
    env = torchax.default_env()
    
    # 创建设备网格用于分布式计算
    # 网格形状为 (设备数量,)，轴名称为 'axis'
    mesh = jax.make_mesh((len(jax.devices()), ), (axis, ))
    env.default_device_or_sharding = NamedSharding(mesh, P())

    # VAE 编译选项：只编译 decode 方法
    vae_options = torchax.CompileOptions(
        methods_to_compile=['decode']
    )
    
    # 将各个模块转换为 JAX 格式并编译
    _move_module(pipe.vae)
    pipe.vae = torchax.compile(pipe.vae)
    
    _move_module(pipe.text_encoder)
    pipe.text_encoder = torchax.compile(pipe.text_encoder)

    # Transformer 需要特殊处理
    _move_module(pipe.transformer)
    # rope.freqs 不是参数或缓冲区，需要手动转换
    pipe.transformer.rope.freqs = pipe.transformer.rope.freqs.to('jax')
    
    # Transformer 编译选项：指定静态参数名称
    options = torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('return_dict',)}
    )
    pipe.transformer = torchax.compile(pipe.transformer, options)

    print('设备数量:', len(jax.devices()))

    # 应用权重分片策略到各个模块
    # 这将模型权重分布到多个设备上以实现并行计算
    
    # 分片 Transformer 权重
    pipe.transformer.params = _shard_weight_dict(
        pipe.transformer.params, 
        transformer_shardings,
        mesh
    )
    pipe.transformer.buffers = _shard_weight_dict(
        pipe.transformer.buffers, 
        transformer_shardings,
        mesh
    )
    
    # 分片文本编码器权重
    pipe.text_encoder.params = _shard_weight_dict(
        pipe.text_encoder.params, 
        text_encoder_shardings,
        mesh
    )
    pipe.text_encoder.buffers = _shard_weight_dict(
        pipe.text_encoder.buffers, 
        text_encoder_shardings,
        mesh
    )

    # VAE 权重完全复制（不分片）
    # 注意：这会在所有设备上复制 VAE
    pipe.vae.params = _shard_weight_dict(pipe.vae.params, {}, mesh)
    pipe.vae.buffers = _shard_weight_dict(pipe.vae.buffers, {}, mesh)

    def move_scheduler(scheduler):
        """
        将调度器的张量移动到 JAX 设备
        （当前已注释，可能不需要）
        """
        for k, v in scheduler.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(scheduler, k, v.to('jax'))

    # move_scheduler(pipe.scheduler)  # 已注释

    def module_size(module):
        """
        计算模块的内存大小（以字节为单位）
        """
        size = 0
        for k, v in module.state_dict().items():
            size += math.prod(v.shape) * v.dtype.itemsize
        return size

    # 打印各个模块的内存使用情况
    for m in dir(pipe):
        module = getattr(pipe, m, None)
        if isinstance(module, torch.nn.Module):
            print(f'{m}: {module_size(module) / (1024 * 1024 * 1024):.2f} GB')

    # 定义生成提示词
    prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
    
    # 负面提示词（指定不想要的内容）
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    # 长提示词示例（海岸风景）
    long_prompt = "Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach.The crashing blue waters create white-tipped waves,while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and greenshrubbery covers the cliffs edge. The steep drop from the road down to the beach is adramatic feat, with the cliff's edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway."

    # 在设备网格上下文中运行生成
    with mesh:
        outputs = []
        # 运行 4 次生成以测试性能和一致性
        for i in range(2):
            start = time.perf_counter()
            
            # 在第 3 次迭代时开始性能分析
            if i == 1:
                jax.profiler.start_trace('/tmp/tensorboard')
            
            # 执行视频生成
            output = pipe(
                prompt=long_prompt,
                negative_prompt=negative_prompt,
                # 低分辨率选项（已注释）
                # height=384,
                # width=640,
                num_inference_steps=50,  # 推理步数
                height=720,              # 视频高度
                width=1280,              # 视频宽度
                # num_frames=81,         # 更多帧数选项（已注释）
                num_frames=41,           # 视频帧数
                guidance_scale=5.0,      # 引导强度
            ).frames[0]
            end = time.perf_counter()
            # 在第 4 次迭代时停止性能分析
            if i == 1:                
                jax.profiler.stop_trace()

            # 打印每次迭代的耗时
            print(f'第 {i} 次迭代: {end - start:.6f} 秒')
            outputs.append(output)

    # 导出第一个生成的视频
    start = time.perf_counter()
    export_to_video(outputs[0], "output.mp4", fps=8)
    end = time.perf_counter()
    print(f'导出视频耗时: {end - start:.6f} 秒')
    print('完成！')

    # 注释：生成视频时长计算公式
    # 视频时长 = (帧数-1)/fps
    # 例如：针对1.3B模型生成5秒视频 = (41-1)/8 = 5秒


if __name__ == '__main__':
    main()