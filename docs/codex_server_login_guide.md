# Codex + 服务器联合使用教程

更新时间：2026-03-11

## 适用场景

这份文档用于本机通过 Codex 操作远端服务器，例如：

- 在本地 Codex 里直接执行远端命令
- 让 Codex 帮忙排查服务器状态
- 后续服务器 IP 变动时快速恢复连接

当前已验证可连接的服务器信息：

- 用户：`hex`
- IP：`114.212.165.225`
- 登录命令：`ssh hex@114.212.165.225`

## 1. 登录教程

### 1.1 先在终端手动验证 SSH

在本机终端执行：

```bash
ssh hex@114.212.165.225
```

首次连接如果出现主机指纹确认，输入：

```text
yes
```

连接成功后，建议先检查这几个信息：

```bash
hostname
whoami
pwd
```

当前这台机器返回过：

- `hostname`: `amax`
- `whoami`: `hex`
- `pwd`: `/home/hex`

### 1.2 推荐配置 `~/.ssh/config`

为了后续让终端和 Codex 都直接使用同一个别名，建议在本机的 `~/.ssh/config` 里添加：

```sshconfig
Host amax
  HostName 114.212.165.225
  User hex
  ServerAliveInterval 60
  ServerAliveCountMax 3
```

保存后可以直接用：

```bash
ssh amax
```

这样做的好处是：

- 后续 IP 变了，只改一处
- 终端和 Codex 都能复用同一个地址别名
- 少打用户名和 IP，不容易输错

### 1.3 在 Codex 里怎么用

如果 Codex 运行在你的本机，它本质上也是通过本机环境执行命令，所以可以直接让 Codex 运行：

```bash
ssh amax
```

或者直接运行单条远端命令：

```bash
ssh amax 'hostname && whoami && nvidia-smi -L'
```

更适合让 Codex 执行的是这种非交互命令，例如：

```bash
ssh amax 'df -h /home'
ssh amax 'quota -s'
ssh amax 'ps -eo pid,user,%cpu,%mem,args --sort=-%cpu | head'
```

如果需要持续交互，优先自己先开一个本地终端连上：

```bash
ssh amax
```

然后再让 Codex 帮你整理命令、解释输出、生成脚本。

## 2. 如果 IP 变化怎么办

### 2.1 最推荐的处理方式

如果服务器 IP 改了，不要到处改命令，直接修改本机 `~/.ssh/config` 里的这一行：

```sshconfig
HostName 114.212.165.225
```

改成新 IP，例如：

```sshconfig
HostName NEW_SERVER_IP
```

这样以后仍然用：

```bash
ssh amax
```

Codex 里也仍然用：

```bash
ssh amax 'your command'
```

### 2.2 如果出现主机指纹冲突

服务器换 IP 或重装系统后，常见报错是：

```text
WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!
```

这是因为本机 `known_hosts` 里还记着旧机器的指纹。处理方法：

```bash
ssh-keygen -R 114.212.165.225
```

如果你平时主要用别名，也可以删别名对应记录：

```bash
ssh-keygen -R amax
```

然后重新连接：

```bash
ssh amax
```

首次重连时再次确认新指纹即可。

### 2.3 如果是学校/机房白名单导致登不上

有些服务器不是单纯改 IP，而是：

- 服务器出口 IP 变了
- 你本机公网 IP 变了
- 机房按 IP 白名单放行

这种情况下需要区分两类变化：

#### 情况 A：服务器 IP 变了

你需要做的是：

- 向管理员确认新服务器 IP
- 修改 `~/.ssh/config` 中的 `HostName`
- 如有旧指纹冲突，执行 `ssh-keygen -R`

#### 情况 B：你本机公网 IP 变了

你需要做的是：

- 先在本机查询当前公网 IP
- 把新公网 IP 发给服务器管理员加入白名单

本机可用下面任一命令查看公网 IP：

```bash
curl ifconfig.me
```

或：

```bash
curl https://api.ipify.org
```

如果管理员已经把你旧 IP 加白，而你换了网络，比如：

- 换了 Wi-Fi
- 回学校或回家
- 手机热点切换

那么就需要重新报备新的公网 IP。

## 3. 推荐的日常使用方式

### 3.1 平时固定用 SSH 别名

统一使用：

```bash
ssh amax
```

不要每次都手打：

```bash
ssh hex@114.212.165.225
```

### 3.2 让 Codex 优先执行单条远端命令

例如：

```bash
ssh amax 'nvidia-smi'
ssh amax 'df -h /home'
ssh amax 'du -sh /home/hex'
```

这种方式最稳定，输出也最容易让 Codex 继续分析。

### 3.3 把常用检查收敛成固定命令

建议记住这几个：

```bash
ssh amax 'hostname && whoami && pwd'
ssh amax 'uptime'
ssh amax 'free -h'
ssh amax 'df -h /home'
ssh amax 'quota -s'
ssh amax 'nvidia-smi'
```

## 4. 一份最小可复用配置

### 本机 `~/.ssh/config`

```sshconfig
Host amax
  HostName 114.212.165.225
  User hex
  ServerAliveInterval 60
  ServerAliveCountMax 3
```

### 连接测试

```bash
ssh amax 'hostname && whoami && pwd'
```

### IP 变化后的修复顺序

1. 确认是服务器 IP 变了，还是你本机公网 IP 变了。
2. 如果是服务器 IP 变了，修改 `~/.ssh/config` 的 `HostName`。
3. 如果报主机指纹冲突，执行 `ssh-keygen -R 旧IP`。
4. 如果是白名单问题，把你当前公网 IP 发给管理员。
5. 最后重新测试：

```bash
ssh amax 'hostname'
```
