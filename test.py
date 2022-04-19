from sentence_transformers.cross_encoder import CrossEncoder

model_save_path = ''

model = CrossEncoder(model_save_path, device="cpu", max_length=64)


with open("test.txt","r")as f, open("result.txt","w")as fw:
    for i, line in enumerate(f):
        if i%1000 == 0:print(i)
        line_lst = line.strip().split("\t")
        score = model.predict([line_lst])[0]
        fw.write(line.strip()+"\t"+str(score)+"\n")
