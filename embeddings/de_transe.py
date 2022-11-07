import torch
import torch.nn as nn

from embedding import Embedding


class DETransE(Embedding):
    def __init__(self, args):
        super(DETransE, self).__init__(args)
        self.args = args

        self.ent_embs = nn.Embedding(args.dataset.numEnt(), args.s_emb_dim).cuda()
        self.rel_embs = nn.Embedding(args.dataset.numRel(), args.s_emb_dim + args.t_emb_dim).cuda()

        self.create_embeddings()

        self.time_nl = torch.sin

        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)

        self.sigm = torch.nn.Sigmoid()
        self.tanh = nn.Tanh()

    def create_embeddings(self):
        self.m_freq = nn.Embedding(self.args.dataset.numEnt(), self.args.t_emb_dim).cuda()
        self.d_freq = nn.Embedding(self.args.dataset.numEnt(), self.args.t_emb_dim).cuda()
        self.y_freq = nn.Embedding(self.args.dataset.numEnt(), self.args.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_freq.weight)
        nn.init.xavier_uniform_(self.d_freq.weight)
        nn.init.xavier_uniform_(self.y_freq.weight)

        self.m_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_phi.weight)
        nn.init.xavier_uniform_(self.d_phi.weight)
        nn.init.xavier_uniform_(self.y_phi.weight)

        self.m_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_amp.weight)
        nn.init.xavier_uniform_(self.d_amp.weight)
        nn.init.xavier_uniform_(self.y_amp.weight)

    def get_time_embeddings(self, entities, year, month, day):
        y = self.y_amp(entities) * self.time_nl(self.y_freq(entities) * year + self.y_phi(entities))
        m = self.m_amp(entities) * self.time_nl(self.m_freq(entities) * month + self.m_phi(entities))
        d = self.d_amp(entities) * self.time_nl(self.d_freq(entities) * day + self.d_phi(entities))

        return y + m + d

    def get_embeddings(self, heads, rels, tails, years, months, days, intervals=None):
        years = years.view(-1, 1)
        months = months.view(-1, 1)
        days = days.view(-1, 1)

        h, r, t = self.ent_embs(heads), self.rel_embs(rels), self.ent_embs(tails)

        h_t = self.get_time_embedd(heads, years, months, days)
        t_t = self.get_time_embedd(tails, years, months, days)

        h = torch.cat((h, h_t), 1)
        t = torch.cat((t, t_t), 1)
        return h, r, t

    def forward(self, heads, rels, tails, years, months, days):
        h_embs, r_embs, t_embs = self.getEmbeddings(heads, rels, tails, years, months, days)

        scores = h_embs + r_embs - t_embs
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = -torch.norm(scores, dim=1)
        return scores