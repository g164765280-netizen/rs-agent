from mentor import MentorAgent, TEACH_SYSTEM
a = MentorAgent()
ctx = '用户知识水平：研一新生，只学过深度学习基础，没接触遥感。当前目标：理解概念。用户问题：变化检测和普通图像分割有什么区别？请按教学原则回答，结尾给出可执行下一步。'
print(a._chat(TEACH_SYSTEM, ctx))
