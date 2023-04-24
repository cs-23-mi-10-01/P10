import torch

from rank.TimePlex.pairwise.gadgets import Recurrent, Pairs

from rank.TimePlex.models_helper import *

def modify_args_timeplex_base(args):
    args["has_cuda"] = False

    return args

time_index = {"t_s": 0, "t_s_orig": 1, "t_e": 2, "t_e_orig": 3, "t_str": 4, "t_i": 5}

class TimePlex_base(torch.nn.Module):
    def __init__(self, entity_count, relation_count, timeInterval_count, embedding_dim, clamp_v=None, reg=2,
                 batch_norm=False, unit_reg=False, normalize_time=True, init_embed=None, time_smoothing_params=None, flag_add_reverse=0,
                 has_cuda=True, time_reg_wt = 0.0, emb_reg_wt=1.0,  srt_wt=1.0, ort_wt=1.0, sot_wt=0.0):

        super(TimePlex_base, self).__init__()
        
        # self.flag_add_reverse = flag_add_reverse
        # if self.flag_add_reverse==1:
        #     relation_count*=2    

        if init_embed is None:
            init_embed = {}
            for embed_type in ["E_im", "E_re", "R_im", "R_re", "T_im", "T_re"]:
                init_embed[embed_type] = None
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.timeInterval_count = timeInterval_count

        self.has_cuda = has_cuda

        self.E_im = torch.nn.Embedding(self.entity_count, self.embedding_dim) if init_embed["E_im"] is None else \
            init_embed["E_im"]
        self.E_re = torch.nn.Embedding(self.entity_count, self.embedding_dim) if init_embed["E_re"] is None else \
            init_embed["E_re"]

        self.R_im = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim) if init_embed["R_im"] is None else \
            init_embed["R_im"]
        self.R_re = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim) if init_embed["R_re"] is None else \
            init_embed["R_re"]

        # E embeddingsfor (s,r,t) and (o,r,t) component
        self.E2_im = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.E2_re = torch.nn.Embedding(self.entity_count, self.embedding_dim)

        # R embeddings for (s,r,t) component
        self.Rs_im = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim)
        self.Rs_re = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim)

        # R embeddings for (o,r,t) component
        self.Ro_im = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim)
        self.Ro_re = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim)

        # time embeddings for (s,r,t)
        self.Ts_im = torch.nn.Embedding(self.timeInterval_count + 2,
                                        self.embedding_dim)  # if init_embed["T_im"] is None else init_embed["T_im"] #padding for smoothing: 1 for start and 1 for end
        self.Ts_re = torch.nn.Embedding(self.timeInterval_count + 2,
                                        self.embedding_dim)  # if init_embed["T_re"] is None else init_embed["T_re"]#padding for smoothing: 1 for start and 1 for end

        # time embeddings for (o,r,t)
        self.To_im = torch.nn.Embedding(self.timeInterval_count + 2,
                                        self.embedding_dim)  # if init_embed["T_im"] is None else init_embed["T_im"] #padding for smoothing: 1 for start and 1 for end
        self.To_re = torch.nn.Embedding(self.timeInterval_count + 2,
                                        self.embedding_dim)  # if init_embed["T_re"] is None else init_embed["T_re"]#padding for smoothing: 1 for start and 1 for end

        ##
        self.pad_max = torch.tensor([timeInterval_count + 1])
        self.pad_min = torch.tensor([0])
        if self.has_cuda:
            self.pad_max = self.pad_max.cuda()
            self.pad_min = self.pad_min.cuda()

        # '''
        torch.nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_im.weight.data, 0, 0.05)

        torch.nn.init.normal_(self.E2_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.E2_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.Rs_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.Rs_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.Ro_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.Ro_im.weight.data, 0, 0.05)

        # init time embeddings
        torch.nn.init.normal_(self.Ts_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.Ts_im.weight.data, 0, 0.05)

        torch.nn.init.normal_(self.To_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.To_im.weight.data, 0, 0.05)
        # '''

        self.minimum_value = -self.embedding_dim * self.embedding_dim
        self.clamp_v = clamp_v

        self.unit_reg = unit_reg

        self.reg = reg
        # print("Regularization value: in time_complex_fast: ", reg)

        self.normalize_time = normalize_time

        self.batch_norm = batch_norm

        # print("batch_norm not being used")

        # --srt, ort weights --#
        self.srt_wt = srt_wt 
        self.ort_wt = ort_wt 
        self.sot_wt = sot_wt

        self.time_reg_wt = time_reg_wt
        self.emb_reg_wt = emb_reg_wt

    def forward(self, s, r, o, t, flag_debug=0):
        if t is not None:
            # if not t.shape[-1]==1:
            if (t.shape[-1] == len(time_index)):  # pick which dimension to index
                t = t[:, :, time_index["t_s"]]
            else:
                t = t[:, time_index["t_s"], :]

        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)
        r_im = self.R_im(r) if r is not None else self.R_im.weight.unsqueeze(0)
        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)
        r_re = self.R_re(r) if r is not None else self.R_re.weight.unsqueeze(0)
        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)

        # embeddings for s,r,t component
        rs_im = self.Rs_im(r) if r is not None else self.Rs_im.weight.unsqueeze(0)
        rs_re = self.Rs_re(r) if r is not None else self.Rs_re.weight.unsqueeze(0)

        # embeddings for o,r,t component
        ro_im = self.Ro_im(r) if r is not None else self.Ro_im.weight.unsqueeze(0)
        ro_re = self.Ro_re(r) if r is not None else self.Ro_re.weight.unsqueeze(0)

        '''
		##added extra 2 embeddings (padding) for semless time smoothing 
		Need to remove those extra embedding while calculating scores for all posibble time points
		##Currenty there is a minor bug in code -- time smoothing may not work properly until you add 1 to all i/p time points
		as seen tim tim_complex_smooth model --Resolved --underflow padding is pad_max and overflow padding is pad_max+1
		'''
        t_re = self.Ts_re(t) if t is not None else self.Ts_re.weight.unsqueeze(0)[:, :-2, :]
        t_im = self.Ts_im(t) if t is not None else self.Ts_im.weight.unsqueeze(0)[:, :-2, :]

        t2_re = self.To_re(t) if t is not None else self.To_re.weight.unsqueeze(0)[:, :-2, :]
        t2_im = self.To_im(t) if t is not None else self.To_im.weight.unsqueeze(0)[:, :-2, :]


        # if flag_debug:
        #     print("Time embedd data")
        #     print("t_re", t_re.shape, torch.mean(t_re), torch.std(t_re))
        #     print("t_im", t_im.shape, torch.mean(t_im), torch.std(t_im))

        #########

        #########
        # '''

        if t is None:
            ##start time scores
            srt = complex_3way_simple(s_re, s_im, rs_re, rs_im, t_re, t_im)
            # ort = complex_3way_simple(o_re, o_im, ro_re, ro_im, t_re, t_im)
            ort = complex_3way_simple(t_re, t_im, ro_re, ro_im, o_re, o_im)

            sot = complex_3way_simple(s_re, s_im,  t_re, t_im, o_re, o_im)

            score = self.srt_wt * srt + self.ort_wt * ort + self.sot_wt * sot

            # --for inverse facts--#
            r = r + self.relation_count / 2
            
            rs_re = self.Rs_re(r)
            rs_im = self.Rs_im(r)
            ro_re = self.Ro_re(r)
            ro_im = self.Ro_im(r)

            srt = complex_3way_simple(o_re, o_im, rs_re, rs_im, t_re, t_im)
            ort = complex_3way_simple(t_re, t_im, ro_re, ro_im, s_re, s_im)
            sot = complex_3way_simple(o_re, o_im,  t_re, t_im, s_re, s_im)

            score_inv = self.srt_wt * srt + self.ort_wt * ort + self.sot_wt * sot
            # ------------------- #
            
            # result = score
            result = score + score_inv

            return result


        if s is not None and o is not None and s.shape == o.shape:  # positive samples
            sro = complex_3way_simple(s_re, s_im, r_re, r_im, o_re, o_im)

            srt = complex_3way_simple(s_re, s_im, rs_re, rs_im, t_re, t_im)

            # ort = complex_3way_simple(o_re, o_im, ro_re, ro_im, t_re, t_im)
            ort = complex_3way_simple(t_re, t_im, ro_re, ro_im, o_re, o_im)

            # sot = complex_3way_simple(s_re, s_im,  t2_re, t2_im, o_re, o_im)
            sot = complex_3way_simple(s_re, s_im,  t_re, t_im, o_re, o_im)

        else:
            sro = complex_3way_fullsoftmax(s, r, o, s_re, s_im, r_re, r_im, o_re, o_im, self.embedding_dim)
            
            srt = complex_3way_fullsoftmax(s, r, t, s_re, s_im, rs_re, rs_im, t_re, t_im, self.embedding_dim)
            
            # ort = complex_3way_fullsoftmax(o, r, t, o_re, o_im, ro_re, ro_im, t_re, t_im, self.embedding_dim)
            ort = complex_3way_fullsoftmax(t, r, o, t_re, t_im, ro_re, ro_im, o_re, o_im, self.embedding_dim)

            # sot = complex_3way_fullsoftmax(s, t, o, s_re, s_im, t2_re, t2_im, o_re, o_im,  self.embedding_dim)
            sot = complex_3way_fullsoftmax(s, t, o, s_re, s_im, t_re, t_im, o_re, o_im,  self.embedding_dim)


        result = sro + self.srt_wt * srt + self.ort_wt * ort + self.sot_wt * sot
        # result = srt

        return result

    def regularizer(self, s, r, o, t, reg_val=0):
        if t is not None:
            # if not t.shape[-1]==1:
            if (t.shape[-1] == len(time_index)):  # pick which dimension to index
                t = t[:, :, time_index["t_s"]]
            else:
                t = t[:, time_index["t_s"], :]

            # if (t.shape[-1] == len(time_index)):  # pick which dimension to index
            #     t = t[:, :, 0]
            # else:
            #     t = t[:, 0, :]

        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)

        ts_re = self.Ts_re(t)
        ts_im = self.Ts_im(t)
        to_re = self.To_re(t)
        to_im = self.To_im(t)

        ####
        s2_im = self.E2_im(s)
        s2_re = self.E2_re(s)
        o2_im = self.E2_im(o)
        o2_re = self.E2_re(o)

        rs_re = self.Rs_re(r)
        rs_im = self.Rs_im(r)
        ro_re = self.Ro_re(r)
        ro_im = self.Ro_im(r)

        ####

        # te_re = self.Te_re(t)
        # te_im = self.Te_im(t)
        if reg_val:
            self.reg = reg_val
        # print("CX reg", reg_val)

        #--time regularization--#
        time_reg = 0.0
        if self.time_reg_wt!=0:
            ts_re_all = (self.Ts_re.weight.unsqueeze(0))#[:, :-2, :])
            ts_im_all = (self.Ts_im.weight.unsqueeze(0))#[:, :-2, :])
            to_re_all = (self.To_re.weight.unsqueeze(0))#[:, :-2, :])
            to_im_all = (self.To_im.weight.unsqueeze(0))#[:, :-2, :])
            
            time_reg = time_regularizer(ts_re_all, ts_im_all) + time_regularizer(to_re_all, to_im_all) 
            time_reg *= self.time_reg_wt
        
        # ------------------#

        if self.reg == 2:
            # return (s_re**2+o_re**2+r_re**2+s_im**2+r_im**2+o_im**2 + tr_re**2 + tr_im**2).sum()
            # return (s_re**2+o_re**2+r_re**2+s_im**2+r_im**2+o_im**2).sum() + (tr_re**2 + tr_im**2).sum()
            rs_sum = (rs_re ** 2 + rs_im ** 2).sum()
            ro_sum = (ro_re ** 2 + ro_im ** 2).sum()
            o2_sum = (o2_re ** 2 + o2_im ** 2).sum()
            s2_sum = (s2_re ** 2 + s2_im ** 2).sum()

            ts_sum = (ts_re ** 2 + ts_im ** 2).sum()
            to_sum = (to_re ** 2 + to_im ** 2).sum()


            ret = (s_re ** 2 + o_re ** 2 + r_re ** 2 + s_im ** 2 + r_im ** 2 + o_im ** 2).sum() + ts_sum + to_sum + rs_sum + ro_sum
            ret = self.emb_reg_wt * (ret/ s.shape[0])


        elif self.reg == 3:
            factor = [torch.sqrt(s_re ** 2 + s_im ** 2), 
                      torch.sqrt(o_re ** 2 + o_im ** 2),
                      torch.sqrt(r_re ** 2 + r_im ** 2),
                      torch.sqrt(rs_re ** 2 + rs_im ** 2),
                      torch.sqrt(ro_re ** 2 + ro_im ** 2), 
                      torch.sqrt(ts_re ** 2 + ts_im ** 2),
                      torch.sqrt(to_re ** 2 + to_im ** 2)]
            factor_wt = [1, 1, 1, 1, 1, 1, 1]
            reg = 0
            for ele,wt in zip(factor,factor_wt):
                reg += wt* torch.sum(torch.abs(ele) ** 3)
            ret =  self.emb_reg_wt * (reg / s.shape[0])
        else:
            # print("Unknown reg for complex model")
            assert (False)

        return ret + time_reg


    def normalize_complex(self, T_re, T_im):
        with torch.no_grad():
            re = T_re.weight
            im = T_im.weight
            norm = re ** 2 + im ** 2
            T_re.weight.div_(norm)
            T_im.weight.div_(norm)

        return

    def post_epoch(self):
        if (self.normalize_time):
            with torch.no_grad():
                # normalize Tr
                # self.normalize_complex(self.Tr_re, self.Tr_im)
                # norm=torch.sqrt(self.Tr_re.weight**2 + self.Tr_im.weight**2)
                # self.Tr_re.weight.div_(norm)
                # self.Tr_im.weight.div_(norm)

                # self.Tr_re.weight.div_(torch.norm(self.Tr_re.weight, dim=-1, keepdim=True))
                # self.Tr_im.weight.div_(torch.norm(self.Tr_im.weight, dim=-1, keepdim=True))

                self.Ts_re.weight.div_(torch.norm(self.Ts_re.weight, dim=-1, keepdim=True))
                self.Ts_im.weight.div_(torch.norm(self.Ts_im.weight, dim=-1, keepdim=True))
                self.To_re.weight.div_(torch.norm(self.To_re.weight, dim=-1, keepdim=True))
                self.To_im.weight.div_(torch.norm(self.To_im.weight, dim=-1, keepdim=True))

        # normalize Te
        # self.normalize_complex(self.Te_re, self.Te_im)
        # self.Te_re.weight.div_(torch.norm(self.Te_re.weight, dim=-1, keepdim=True))
        # self.Te_im.weight.div_(torch.norm(self.Te_im.weight, dim=-1, keepdim=True))

        if (self.unit_reg):
            self.E_im.weight.data.div_(self.E_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_re.weight.data.div_(self.E_re.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_im.weight.data.div_(self.R_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_re.weight.data.div_(self.R_re.weight.data.norm(2, dim=-1, keepdim=True))
        return ""

class TimePlex(torch.nn.Module):
    def __init__(self, entity_count, relation_count, timeInterval_count, embedding_dim, batch_norm=False, reg=2,
                 train_kb=None,
                 has_cuda=True, freeze_weights=True,
                 model_path="", recurrent_args={}, pairs_args={}, pairs_wt=0.0, recurrent_wt=0.0, eval_batch_size=10, use_obj_scores=True, 
                 srt_wt=1.0, ort_wt=1.0, sot_wt=0.0, base_model_inverse=False):
        super(TimePlex, self).__init__()

        self.entity_count = entity_count
        self.relation_count = relation_count

        # print("Recurrent args:",recurrent_args)
        # print("Pairs args:",pairs_args)

        # --Load pretrained TimePlex(base) embeddings--#
        if model_path != "":
            # print("Loading embeddings from model saved at {}".format(model_path))
            state = torch.load(model_path, map_location="cpu")
            model_arguments = modify_args_timeplex_base(state['model_arguments'])
            self.base_model = TimePlex_base(**model_arguments)
            self.base_model.load_state_dict(state['model_weights'])
            base_model_inverse = state['model_arguments'].get('flag_add_reverse',False)
            # print("Initialized base model (TimePlex)")

        else:
            raise Exception("Please provide path to Timeplex(base) embeddings")
        # ----------#

        self.embedding_dim = embedding_dim

        self.base_model_inverse = base_model_inverse
        # print("***Base model inverse:{}".format(self.base_model_inverse))


        # --Freezing base model--#
        # '''
        if freeze_weights:
            # print("Freezing base model weights")
            for param in self.base_model.parameters():
                param.requires_grad = False
        # else:
        #     print("Not freezing base model weights")

        self.freeze_weights = freeze_weights
        # '''
        # ----------------------#

        self.minimum_value = -(embedding_dim * embedding_dim)

        self.pairs_wt = pairs_wt
        self.recurrent_wt = recurrent_wt

        if pairs_wt!=0.0:
            self.pairs = Pairs(train_kb, entity_count, relation_count, load_to_gpu=has_cuda,
                                                eval_batch_size=eval_batch_size,
                                                use_obj_scores=use_obj_scores, **pairs_args)
            # print("Initialized Pairs")
        # else:
        #     print("Not  Initializing Pairs")



        if recurrent_wt!=0.0:
            self.recurrent = Recurrent(train_kb, entity_count, relation_count, load_to_gpu=has_cuda,
                                                eval_batch_size=eval_batch_size,
                                                use_obj_scores=use_obj_scores, **recurrent_args)
        #     print("Initialized Recurrent")

        # else:
        #     print("Not Initializing Recurrent")
                    
        # pdb.set_trace()

    def forward(self, s, r, o, t, flag_debug=False):

        # if not self.base_model_inverse:
        if not self.base_model_inverse or t is None:
            base_score = self.base_model(s, r, o, t)
        else:
            rel_cnt = self.relation_count
            if s is None:
                base_score = self.base_model(o, r + rel_cnt, s, t)
            elif o is None:
                base_score = self.base_model(s, r, o, t)
            else:
                base_score = self.base_model(s, r, o, t) + self.base_model(o, r + rel_cnt, s, t)

        pairs_score = self.pairs(s, r, o, t) if self.pairs_wt else 0.0
        recurrent_score = self.recurrent(s, r, o, t) if self.recurrent_wt else 0.0
                
        return base_score + self.pairs_wt * pairs_score + self.recurrent_wt * recurrent_score


    def post_epoch(self):
        base_post_epoch = self.base_model.post_epoch()
        return base_post_epoch

    def regularizer(self, s, r, o, t=None):
        pairs_reg = self.pairs.regularizer(s, r, o, t) if self.pairs_wt else 0.0
        recurrent_reg = self.recurrent.regularizer(s, r, o, t) if self.recurrent_wt else 0.0
        
        # pdb.set_trace()

        if self.freeze_weights:
            return pairs_reg + recurrent_reg
        else:
            return pairs_reg + recurrent_reg + self.base_model.regularizer(s, r, o, t)

