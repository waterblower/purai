# TinyStories 10 MB：实测结果

训练输入 10,023,508 字节（11,459 篇故事及空行分隔符），独立官方验证集 1,000,375 字节（1,286 篇）。
使用固定文件前缀子集，非全量数据；实验协议和复现命令见 [TINYSTORIES.md](../../TINYSTORIES.md)。

## 验证集结果

BPB 是按 UTF-8 字节加权的负对数概率，越低越好。n-gram 从 0–10 阶中比较；本验证集也用于展示逐轮指标，因此这些是验证结果，不是独立最终测试集结果。

| 模型 | bits/byte | 上下文数量 | 计数步数/字节 |
|---|---:|---:|---:|
| n-gram，8 字节上下文 | 1.314825 | 1,229,742（该阶） | — |
| AFRN 默认 | 1.344683 | 103,981 | 27.2 |
| AFRN 默认，关闭 match | 1.375559 | 103,981 | 26.2 |
| AFRN 关闭 Fold | 1.343478 | 103,853 | 27.7 |
| AFRN 关闭 Fold，关闭 match | 1.374359 | 103,853 | 26.7 |

步数是解释器定义的操作计数，未计入全部标量运算，不等于 CPU 指令数。n-gram 表中数量仅为选中阶的上下文数，实际预测也使用低阶表。

## Fold 对照

关闭 Fold − 默认模型 = -0.001205 bits/byte；正值表示 Fold 更好。
按验证故事配对重采样的 95% 百分位区间为 [-0.001661, -0.000757]（2,000 次，种子 42）。
该区间仅描述固定模型在这些验证故事上的抽样波动，不包含更换训练子集带来的不确定性。默认压缩对两组分别执行，因此这是完整训练配置的对照。

## 模型与运行

- `default.data`：2,241,880 字节；训练 57.84 分钟；结构 {"ctxs": 103981, "seqs": 2000, "alts": 679, "tokens": 10023508}。
- `no-fold.data`：2,183,912 字节；训练 49.15 分钟；结构 {"ctxs": 103853, "seqs": 2000, "alts": 0, "tokens": 10023508}。

两组训练并行运行，耗时受资源竞争影响。模型源码未改动；仅新增数据准备、实验与报告脚本。

## 固定提示词的全部生成样本

每个样本生成 600 字节，随机种子 42；以下包含所有预设提示词与温度，没有根据流畅程度筛选。

### default，温度 0.8

```text
Once upon a time, there was a little girl named Lily. She saw a special or name was so glad the end of block. Once upon a time, there was a little girl named Lily. She saw a special box and waved good friends. One day, Timmy were two friends. The bear!” she was inside the line and smiled and said, "Lee years old and see what a red and said, "Wow, looking up and the end of fun together, dreamily came to a park. One day, Timmy's dad were always be fun!"
Anna and Ben says, "They are sadly, he found it and made living room, kept flying happy they went to the would because he was free flowers in the soft was so happy.
As they always ready for lunch
```

### default，温度 1

```text
Once upon a time, there was a little girl named Lily. She ran organized that he wanted to be can roll, Lily. While on their dinner. She said, and all the build a new things around in the faithfork hard and she had a picnic in the sky. Chlikes to be goose to her grandma and the dad playing jumped on the air was the ride, there was interesting. One day, he stumbled upon a time, there was a little girl named Lily. She loved to run afterwards the snack. The bird. "The truck candy wished fox noticed a lot of the man who could hear the very brave.
One day in the would believe horse and cheering and run to three years old and stoil table. 
Lily said. "
```

### default，温度 0.8

```text
Tom found a small red box in the garden. One day, Tom and Anna and Ben though she felt embarrashiny and started to weigher and swing and they were not fly or the pot of a suddenly, a chair and see it to see a big tree. The little girl named Lily. She saw a big, open and smiled and said, "Lee years old and see the ball. We careful not most exciting stay together and soon the bench, she climbed on the ball. He said, "Deal."
The monkey and screamed and saw some bird was not dull decided to playing scarf. They made of for being runs too hard for them. The water. They did not put them out to pluck out!" The monkey and Alf on the picture 
```

### default，温度 1

```text
Tom found a small red box in the garden. Lily helped her make race to dress and forest, feeling very and very play with his friends, Billy nods and forest, her and the castle anymore. It was disappeared in a neighbor were could weepin around the group of hair." Joe accidentalk. Lily quickly crayon the pearl!" he called the forest one looked every day, Tom room. His mom asked his mom and down and they both happy too!

Once upon a time there was a farmer. They alone. He grabs the patient. Now you have to be helpful said.
Mum said, "Sure the park. With a yellow squirrel and start to mother shy little girl smiled,

Li He to him to play 
```

### default，温度 0.8

```text
The little rabbit was afraid of the dark. One day, Tom and Anna and Ben though she felt embarrashiny and started to weigher and swing and they were not fly or the pot of a suddenly, a chair and see it to see a big tree. The littling in the world's so special strong and her mom proud of the bad.
The old making lots of toys and having so much funny hopped throw the window. They make sure he would oft the park with the ball were always be fun!"
Anna and Ben though she felt sadly, he found it and made living room, kept flying happy they went to the would because he was do.

Once upon a time, there was the water gun and again and shared s
```

### default，温度 1

```text
The little rabbit was afraid of the dark. Lily helped her make race to dress and forest, feeling very and very play with his friends, Billy nods and forest, her and the castle anymore. It was a little girl called his friend, no, your back to the girl went of this she and his mom like she said yes! But he could shake box!“I will clouds. She says.
She was exhat day on, shirsty. She said Lily. "I'm spoon. Max plants.

Ben were amazed and they could do something in respectfulness. She like she was happy. They like the ground.
Lily and he was so returning was so returned around the time, there?" 
Suddenly, she stocean and ran back to th
```

### no-fold，温度 0.8

```text
Once upon a time, there was a little girl named Lily. She saw a special or name was so happy and some fish water all over, she cried. She showed her that day on, Timmy was dark. He had a famous and not his mom gave him a kisses. He was so hard. The girl named Lily. She had so much fun. He said he had be brave and goodbye to help them each and explore the nuts it. The dolls, but she didn't wanted to dog. He was too late. The other children were play with a perfect doll went to the marble she says, "What are you are very kind of fish with her dolls and proud too. They loved to eat a little girl named Lily. She saw a big hugged the box and shared s
```

### no-fold，温度 1

```text
Once upon a time, there was a little girl named Lily. She ran organized that he wanted to be can roll, Lily. While on their dinner. She said, and all the bugs. Max could creature she shed Tinought he said no. He knew it come over took a phone together." Some now what is sorry.
"Give me cookies and make Lily. She lived in a line brother brother would jumped. They hugged their village with his friend Sarah and they both happy too!

Once upon a time there.

Once upon a time for breakfast wanted to find it" said "Well, I love was scared, but it forgot the people." Tim was so rest believe it wearing her forward, but kitchen. Thank garagon up to the h
```

### no-fold，温度 0.8

```text
Tom found a small red box in the garden. One day, Tom and Anna and Ben though she felt embarrashiny and started to were happy torn. He loved sweet and found a hard. He had a famous and not his mom gave him a kisses. He was so happy. Because the flowers the boy. He like to play with a big field we came to visit. He children in his house. She says.
They hugged the wheels. They make right away.

Once upon a time, there was wet and dad been too closed her eye?" Her money, and yellow apressed he was performed and share. It looked at all. Sara and Tommy, they love you doing here?" asked the sky. He she was very happen. 
"Maybe we came to 
```

### no-fold，温度 1

```text
Tom found a small red box in the garden. Lily helped her make new friends. Once there was a little bird started to you liked to you feeling very pond. He looked night, Lily feels were in the girl named Lily.
"He move there was busy. "Can went to the wind called So, Rach. They like held on a brave and for you?" Sally told her show it. She top of the toy to played with their lesson than all the dog. The man help butter and mom, looking found. Every day, the wind on the kitchen, she wind, and John saw the blocks in Tom's friends. They's audience. You williant shouted at a pieces.
One day, Timmy's mommy asked his Dadd Lily.
"What are yo
```

### no-fold，温度 0.8

```text
The little rabbit was afraid of the dark. One day, Tom and Anna and Ben though she felt embarrashiny and started to were happy torn. He loved sweet and found a careful. He was a witch her mommy said hello. Every day, Lily's mom something sprinkled liked to be night called out of some more. He was a secret with the ball. We carefused to see the world. It was so happy to play with her mom and dad sang the celery important to take their bloud very excited to explained the colder bike with something shiny in the lit up and got a little red parkers. With them how prison his faces!" 
The spin. She saw a picks up into a sack to play with he
```

### no-fold，温度 1

```text
The little rabbit was afraid of the dark. Lily helped her make new friends. Once there was a little.

Anna, there was very goodbye. She loved home. I print their room is a red anymore. It was a little girl called his friend, a journey and swings, his speed across the best pretty flower!" The bugs car disg with the bag. Suddenly, they see the next day, the flower eyes, so she could not each other. They both happy too!

Once upon a time there.

Once upon a time for breakfast wanted to get it felt go off. They like the how to be friends. 
The moral visit away. The tree thanked his mom's house," her more bugirl not a table. 
Lily said. "
```
