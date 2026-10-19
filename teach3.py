from mentor import MentorAgent, TEACH_SYSTEM
a = MentorAgent()
ctx = '用户是研一新生没接触遥感，问：变化检测和图像分割有什么区别？请用3-4小段简洁回答（总览/输入输出区别/难点区别/方法区别），结尾给一句可执行建议。'
print(a._chat(TEACH_SYSTEM, ctx, max_tokens=800))
