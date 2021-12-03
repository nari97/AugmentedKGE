from Models.TransH import TransH
from Models.TransE import TransE
from Models.TransD import TransD
from Models.SimplE import SimplE
from Models.DistMult import DistMult
from Models.ComplEx import ComplEx
from Models.HolE import HolE
from Models.RotatE import RotatE

import torch

torch.manual_seed(2)

data_to_test = {"batch_h" : torch.LongTensor([0,2,4]), "batch_r" : torch.LongTensor([0,1,2]), "batch_t" : torch.LongTensor([1,3,5])}
for mod in ["transe", "transh", "transd", "simple", "complex", "rotate", "hole", "distmult"]:
    print ("========================================")
    print (mod)
    if mod == "transe":
        model = TransE(6, 3, 3, norm = 2, inner_norm= True)

        head_emb = model.get_head_embeddings(data_to_test)
        tail_emb = model.get_tail_embeddings(data_to_test)
        rel_emb = model.get_relation_embeddings(data_to_test)

        h1,t1,r1 = model.inner_normalize(head_emb, tail_emb, rel_emb)
        h2, t2 = model.normalize_inner(head_emb["e"], tail_emb["e"])

        print ("Head embeddings : ", torch.allclose(h1["e"],h2), "Tail embeddings : ", torch.allclose(t1["e"], t2))
        

    elif mod == "transh":
        model = TransH(6, 3, 3, norm = 2, inner_norm= True)

        head_emb = model.get_head_embeddings(data_to_test)
        tail_emb = model.get_tail_embeddings(data_to_test)
        rel_emb = model.get_relation_embeddings(data_to_test)

        h1,t1,r1 = model.inner_normalize(head_emb, tail_emb, rel_emb)
        h2, t2, wr = model.normalize_inner(head_emb["e"], tail_emb["e"], rel_emb["w_r"])

        print ("Head embeddings : ", torch.allclose(h1["e"] , h2), " Tail embeddings : ", torch.allclose(t1["e"], t2), "W_r : ", torch.allclose(r1["w_r"], wr))

    elif mod == "transd":
        model = TransD(6, 3, 3, 2, norm = 2, inner_norm= True)

        head_emb = model.get_head_embeddings(data_to_test)
        tail_emb = model.get_tail_embeddings(data_to_test)
        rel_emb = model.get_relation_embeddings(data_to_test)

        h1,t1,r1 = model.inner_normalize(head_emb, tail_emb, rel_emb)
        h2, r2, t2, ht2, tt2 = model.normalize_inner(head_emb["e"], rel_emb["r"], tail_emb["e"], head_emb["e_t"], tail_emb["e_t"])

        print ("Head embeddings : ", torch.allclose(h1["e"] , h2), " Tail embeddings : ", torch.allclose(t1["e"], t2), "Relations embeddings : ", torch.allclose(r1["r"], r2), " Head transfer embeddings : " ,torch.allclose(h1["e_t"], ht2), "Tail transfer embeddings : ", torch.allclose(t1["e_t"], tt2))
    
    elif mod == "simple":
        # model = SimplE(6, 3, 3)

        # head_emb = model.get_head_embeddings(data_to_test)
        # tail_emb = model.get_tail_embeddings(data_to_test)
        # rel_emb = model.get_relation_embeddings(data_to_test)

        # h1,t1,r1 = model.inner_normalize(head_emb, tail_emb, rel_emb)

        # print (head_emb)
        # print ("Head embeddings : ", torch.allclose(h1["e"] , head_emb["e"]), " Tail embeddings : ", torch.allclose(t1["e"], tail_emb["e"]), "Relations embeddings : ", torch.allclose(r1["r"], rel_emb["r"]))
        pass
    elif mod == "distmult":
        model = DistMult(6, 3, 3, 2, inner_norm= True)

        head_emb = model.get_head_embeddings(data_to_test)
        tail_emb = model.get_tail_embeddings(data_to_test)
        rel_emb = model.get_relation_embeddings(data_to_test)

        h1,t1,r1 = model.inner_normalize(head_emb, tail_emb, rel_emb)
        h2, r2, t2 = model.normalize_inner(head_emb["e"], rel_emb["r"], tail_emb["e"])

        print ("Head embeddings : ", torch.allclose(h1["e"] , h2), " Tail embeddings : ", torch.allclose(t1["e"], t2), "Relations embeddings : ", torch.allclose(r1["r"], r2))

    elif mod == "complex":
        model = ComplEx(6, 3, 3, norm = 2, inner_norm= True)

        head_emb = model.get_head_embeddings(data_to_test)
        tail_emb = model.get_tail_embeddings(data_to_test)
        rel_emb = model.get_relation_embeddings(data_to_test)

        h1,t1,r1 = model.inner_normalize(head_emb, tail_emb, rel_emb)


        print ("Head embeddings : ", torch.allclose(h1["e_real"] , head_emb["e_real"]), torch.allclose(h1["e_img"] , head_emb["e_img"]), " Tail embeddings : ", torch.allclose(t1["e_real"], tail_emb["e_real"]),torch.allclose(t1["e_img"], tail_emb["e_img"]), "Relations embeddings : ", torch.allclose(r1["r_real"], rel_emb["r_real"]), torch.allclose(r1["r_img"], rel_emb["r_img"]))

    elif mod == "rotate":
        model = RotatE(6, 3, 3, norm = 2, inner_norm= True)

        head_emb = model.get_head_embeddings(data_to_test)
        tail_emb = model.get_tail_embeddings(data_to_test)
        rel_emb = model.get_relation_embeddings(data_to_test)

        h1,t1,r1 = model.inner_normalize(head_emb, tail_emb, rel_emb)


        print ("Head embeddings : ", torch.allclose(h1["e"] , head_emb["e"]), " Tail embeddings : ", torch.allclose(t1["e"], tail_emb["e"]), "Relations embeddings : ", torch.allclose(r1["r"], rel_emb["r"]))

    elif mod == "hole":
        model = HolE(6, 3, 3, 2, inner_norm= True)

        head_emb = model.get_head_embeddings(data_to_test)
        tail_emb = model.get_tail_embeddings(data_to_test)
        rel_emb = model.get_relation_embeddings(data_to_test)

        h1,t1,r1 = model.inner_normalize(head_emb, tail_emb, rel_emb)
        h2, r2, t2 = model.normalize_inner(head_emb["e"], rel_emb["r"], tail_emb["e"])

        print ("Head embeddings : ", torch.allclose(h1["e"] , h2), " Tail embeddings : ", torch.allclose(t1["e"], t2), "Relations embeddings : ", torch.allclose(r1["r"], r2))






