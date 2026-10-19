from mentor import MentorAgent
a = MentorAgent(memory_dir='/tmp/mentor_mem')
u = 'u1'
print('=== 第1轮（应问背景）===')
print(a.chat(u, '变化检测用什么指标评估？'))
a.set_background(u, '研一新生，学过DL基础，没接触遥感', '理解概念')
print()
print('=== 第2轮（应带状态直接教学，不再问背景）===')
print(a.chat(u, '变化检测和图像分割有什么区别？')[:400])
