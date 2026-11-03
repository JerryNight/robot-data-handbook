# 录制格式：rosbag2 到底是什么，以及它为什么不能当训练集

> 阶段 3 · 数据流水线 ｜ 写作时间 2026-11
>
> 配套实验：[`examples/03_data_pipeline/01_rosbag2_layout.py`](../examples/03_data_pipeline/01_rosbag2_layout.py)

## 1. 它解决什么问题

你从机器人上抓下来的第一份数据，大概率是一个 `rosbag2` 目录：

```
my_recording/
├── metadata.yaml       # 几千字节，袋子的"封面"
└── bag_0.db3           # 几十 GB，真正的数据
```

新手会把这两个文件当成"视频"或者"数据库"，然后直接拿去训练。**这一步几乎一定会出错**，而且错得不明显——文件能打开、能回放、时间戳连续，就是训练效果莫名其妙地差。

这一篇要把它看清楚：rosbag2 是**录制格式**，设计目标是"在机器人上低延迟、不丢地写盘"，不是"给人高效地读"。理解它的结构，你才知道转换层必须补什么。

## 2. 结构：一个 SQLite 文件，两张表

`.db3` 就是标准的 SQLite 数据库。表结构固定为两张：

```sql
CREATE TABLE topics(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL,
  serialization_format TEXT NOT NULL, offered_qos_profiles TEXT NOT NULL);

CREATE TABLE messages(
  id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL,
  timestamp INTEGER NOT NULL, data BLOB NOT NULL);
```

就这么多。一帧图像是 `messages` 表里的一行，`data` 列是一个 BLOB；`timestamp` 单位是**纳秒**。

所以：

- 图像、关节角、IMU、点云，**全部挤在同一张表里**，按时间交错
- 没有列式存储、没有按 topic 分区、没有二级索引（只有 `id` 主键）
- `data` 列里是什么，取决于 `topics.type`。要解它，得先有消息定义（`.msg`），再按 CDR 规则解字节

实验里生成一个 60 秒、3 个话题的 bag，量级如下：

```
topic                                   Hz        条数          总量
----------------------------------------------------------------
/camera/image_raw/compressed            30      1800     61.08MB
/joint_states                          100      6000      1.03MB
/imu/data                              200     12000      1.58MB

图像：条数占 9.1%，体积占 95.9%
```

**这是第一个必须记住的数字**：图像只占消息条数的 9%，却占了 96% 的字节。这决定了后面所有关于存储、压缩、去重的讨论都是围绕图像展开的。

## 3. metadata.yaml：袋子的"封面"，也是最容易骗人的地方

`metadata.yaml` 记录了每个 topic 多少条、每个分片起止时间、整体时长。rosbag2 靠它做索引，不用打开 `.db3` 就能知道时长和条数。

它的危险在于：**这是自我声明，不是校验过的**。

实验里演示了标准事故：开启 30 秒分片后，目录变成两个文件：

```
bag_0.db3      30.0s     9900 条
bag_1.db3      30.0s     9900 条
```

`metadata.yaml` 老老实实写着 `message_count: 19800`。但如果上传脚本只传了 `bag_0.db3`（比如按文件名 glob 时漏了、或者传到一半超时）：

- `bag_0.db3` **自身完全自洽**：表结构完整、时间戳连续、能正常回放
- 它只是安静地少了 30 秒
- 没有任何一层会报错

这是数据平台最常见的一类事故。判据搞错是根本原因：**"录制成功"的判据是"文件写完了"，不是"数据完整"**。

唯一可靠的校验是拿 metadata 的预期值和实际内容对账：

```python
expected = metadata['message_count']
actual   = cur.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
assert actual == expected, 'bag 不完整，拒绝入库'
```

三行代码，能挡掉一整类事故。**这就是"质检"这一环存在的意义**，也是流水线里最容易被跳过的一环。

## 4. 两种缺陷，两种检测方法

实验在 bag 里埋了两个真实场景的缺陷：

| 缺陷 | 成因 | 检测方法 |
|---|---|---|
| `/joint_states` 在 t=30s 丢 20 帧 | DDS QoS 配错 / 网络拥塞 | 相邻消息间隔 > 1.5 × 标称周期 |
| `/imu/data` 在 t=45s 后时间戳 +50ms | 时钟跳变（NTP 校正、时钟域切换） | 相邻时间戳增量超出容差 |

```
检测 ①：t=29.99s 处空洞 210 ms（正常 10.0ms）-> 丢了约 20 帧
检测 ②：第 8999 到第 9000 条：间隔 55.0 ms（应为 5.0ms）
```

第二种检测有个坑值得单独说：**必须按 `ORDER BY id` 查，不能按 `ORDER BY timestamp`**。时钟跳变之后，按时间排序会把数据顺序搞乱，反而看不出异常。写入顺序（`id`）才是真相。

这两类缺陷的共同点：**都不会让录制失败**。rosbag2 照常写盘、照常关闭、照常返回成功。但它们会变成训练集里 200ms 的动作-观测错配，或者 20 帧的静默丢帧——而训练 loss 曲线看不出来。

## 5. 所以它为什么不能当训练集

不是因为"不好"，是因为设计目标不同。用训练集的眼光要求它是错位的：

| 需求 | rosbag2 的能力 | 训练集需要什么 |
|---|---|---|
| 随机访问单帧 | 差：BLOB 要整行读，无法只取某一维 | 分块布局 / 分片 Parquet |
| 跨 episode 查询 | 差：每个 bag 是独立库，无全局索引 | 数据集清单 + 索引文件 |
| 压缩 | 无（ZSTD 只能压整个文件） | 按模态分别编码 |
| 增量写入 | 好：append 一行即可 | （训练集不需要） |
| 元数据 | 只有 topic 名和类型 | 标定、任务、成功率、语言指令 |
| 版本与血缘 | 无 | 内容哈希 + manifest |

正确的分工是：**rosbag2 负责采集，转换层负责重排成列式 + 索引 + 元数据**。

## 6. 没有 ROS 环境也能读

这一点很实用。因为 `.db3` 就是 SQLite，任何装了 Python 的机器都能读，不需要 ROS、不需要 `rosbag2_py`：

```python
import sqlite3
conn = sqlite3.connect("bag_0.db3")
cur = conn.cursor()
# 每个 topic 的条数与时间跨度
for tid, name in cur.execute("SELECT id, name FROM topics"):
    row = cur.execute(
        "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) "
        "FROM messages WHERE topic_id=?", (tid,)).fetchone()
    print(name, row)
```

做数据平台的第一个工具就该是这个——**在写任何转换代码之前，先能核对原始数据**。实验里的 `query_file()` 就是这样一个 20 行的函数。

同理，`metadata.yaml` 的结构足够简单，不需要 PyYAML 也能解析（固定缩进 + 固定字段）。

## 7. 对数据平台的四条结论

1. **入库前必须对账**。`metadata.message_count` 和 `SELECT COUNT(*)` 必须相等，时间跨度必须等于 `duration`。不通过就拒绝入库，不要"先存着以后修"。

2. **时长的真值是时间跨度，不是文件大小**。60 秒的 bag 因为分片可能只有 30 秒；反过来，一个 30 秒的分片也可能因为重复写入有 40 秒的时间跨度。只有 `MAX(timestamp) - MIN(timestamp)` 说的算。

3. **质检要查的是"没发生的事"**。丢帧、时钟跳变、分片缺失都不会报错，只能主动检测。所以质检规则必须是**白名单式**的：预期什么，就断言什么。

4. **别在原始 bag 上训练**。哪怕只是临时跑个 baseline。因为一旦有人拿 baseline 结果做决策，你就得永远保留这份原始数据不被清理——而它占的空间是转换后的 70 倍。

## 8. 下一步

录制格式解决了"数据怎么进来"。下一件是"数据里的时间到底是哪一刻"——即使 bag 写得很完美，如果时间戳来自系统时钟，`02/time-sync` 里那 6.5ms 的系统性滞后依然存在。

见 [`clock-and-trigger.md`](clock-and-trigger.md)。

---

延伸阅读：
- [02/传感器数据](../02-sensor-data/) —— 每种传感器的数据形态
- [formats.md](formats.md) —— 转换目标格式的取舍
- [MCAP 格式规范](https://mcap.dev/) —— rosbag2 之外的另一种容器选择
