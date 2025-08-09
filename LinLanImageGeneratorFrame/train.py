from .Losses import *

def train_ddpm(dataloader, model: nn.Module, opt: torch.optim, Epoch: int = 100, file_name: str = None,
               device: str = "cuda",
               clip_grad: bool = False):
    scaler = GradScaler()
    for epoch in range(Epoch):
        model.train()
        for index, (tx, *_) in enumerate(tqdm(dataloader, desc=f"{epoch}/{Epoch}")):
            tx = tx.to(device)
            opt.zero_grad()
            with autocast():
                loss = model(tx)
                if torch.isnan(loss).any():
                    print("NaN detected! Skipping step")
                    opt.zero_grad()
                    continue
            scaler.scale(loss).backward()  # 缩放损失并反向传播
            if clip_grad:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)  # 更新参数
            scaler.update()
        if file_name:
            try:
                name = file_name.split('/')[-1].split(".")[0]
                with open(name + ".json", 'w') as f:
                    json.dump(model.config, f)
                torch.save(model.state_dict(), file_name)
            except:
                torch.save(model, file_name)


def train_vae(dataloader, model: nn.Module, opt: torch.optim, loss_func: nn.Module = None, Epoch: int = 100,
              file_name: str = None, device: str = "cuda",
              clip_grad: bool = False):
    if loss_func is None:
        loss_func = VAELoss()
    scaler = GradScaler()
    for epoch in range(Epoch):
        model.train()
        for index, (tx, *_) in enumerate(tqdm(dataloader, desc=f"{epoch}/{Epoch}")):
            tx = tx.to(device)
            opt.zero_grad()
            with autocast():
                out, latent = model(tx)
                loss = loss_func(pre=out, y=tx, latent=latent)
                if torch.isnan(loss).any():
                    print("NaN detected! Skipping step")
                    opt.zero_grad()
                    continue
            scaler.scale(loss).backward()  # 缩放损失并反向传播
            if clip_grad:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)  # 更新参数
            scaler.update()
        if file_name:
            try:
                name = file_name.split('/')[-1].split(".")[0]
                with open(name + ".json", 'w') as f:
                    json.dump(model.config, f)
                torch.save(model.state_dict(), file_name)
            except:
                torch.save(model, file_name)


def train_vqvae(dataloader, model: nn.Module, opt: torch.optim, loss_func: nn.Module = None, Epoch: int = 100,
                file_name: str = None, device: str = "cuda",
                clip_grad: bool = False):
    if loss_func is None:
        loss_func = VQVAELoss()
    scaler = GradScaler()
    for epoch in range(Epoch):
        model.train()
        for index, (tx, *_) in enumerate(tqdm(dataloader, desc=f"{epoch}/{Epoch}")):
            tx = tx.to(device)
            opt.zero_grad()
            with autocast():
                out, book_loss = model(tx)
                loss = loss_func(pre=out, y=tx, book_loss=book_loss)
                if torch.isnan(loss).any():
                    print("NaN detected! Skipping step")
                    opt.zero_grad()
                    continue
            scaler.scale(loss).backward()  # 缩放损失并反向传播
            if clip_grad:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)  # 更新参数
            scaler.update()
        if file_name:
            try:
                name = file_name.split('/')[-1].split(".")[0]
                with open(name + ".json", 'w') as f:
                    json.dump(model.config, f)
                torch.save(model.state_dict(), file_name)
            except:
                torch.save(model, file_name)


def train_dcgan(dataloader, g_model: nn.Module, g_opt: torch.optim, d_model: nn.Module, d_opt: torch.optim,
                loss_func: nn.Module = None, Epoch: int = 100,
                g_file_name: str = None, d_file_name: str = None, device: str = "cuda",
                clip_grad: bool = False, in_dim: int = None, g_model_train_step=5):
    if loss_func is None:
        loss_func = nn.BCELoss()
    if in_dim is None:
        if hasattr(g_model, 'in_dim'):
            in_dim = g_model.in_dim
        else:
            raise ValueError("生成器模型必须定义 'in_dim' 属性或显式传入 in_dim 参数")
    scaler = GradScaler()
    for epoch in range(Epoch):
        g_model.train()
        d_model.train()
        for index, (tx, *_) in enumerate(tqdm(dataloader, desc=f"{epoch}/{Epoch}")):
            tx = tx.to(device)
            norise = torch.randn(size=(tx.shape[0], in_dim)).to(device)
            true_label = torch.ones(size=(tx.shape[0], 1)).to(device)
            false_label = torch.zeros(size=(tx.shape[0], 1)).to(device)
            fake_image = g_model(norise)
            d_opt.zero_grad()
            with autocast():
                dis_true_pre = d_model(tx)
                dis_true_loss = loss_func(dis_true_pre, true_label)
                dis_false_pre = d_model(fake_image.detach())
                dis_false_loss = loss_func(dis_false_pre, false_label)
                d_loss = (dis_true_loss + dis_false_loss) / 2
                if torch.isnan(d_loss).any():
                    print("NaN detected! Skipping step")
                    d_opt.zero_grad()
                    continue
            scaler.scale(d_loss).backward()  # 缩放损失并反向传播
            if clip_grad:
                scaler.unscale_(d_opt)
                torch.nn.utils.clip_grad_norm_(d_model.parameters(), max_norm=1.0)
            scaler.step(d_opt)  # 更新参数
            scaler.update()
            if index % g_model_train_step == 0:
                g_opt.zero_grad()
                with autocast():
                    g_dis_pre = d_model(fake_image)
                    g_loss = loss_func(g_dis_pre, true_label)
                    if torch.isnan(g_loss).any():
                        print("NaN detected! Skipping step")
                        g_opt.zero_grad()
                        continue
                scaler.scale(g_loss).backward()  # 缩放损失并反向传播
                if clip_grad:
                    scaler.unscale_(g_opt)
                    torch.nn.utils.clip_grad_norm_(g_model.parameters(), max_norm=1.0)
                scaler.step(g_opt)  # 更新参数
                scaler.update()
        if g_file_name:
            try:
                name = g_file_name.split('/')[-1].split(".")[0]
                with open(name + ".json", 'w') as f:
                    json.dump(g_model.config, f)
                torch.save(g_model.state_dict(), g_file_name)
            except:
                torch.save(g_model, g_file_name)
        if d_file_name:
            try:
                name = d_file_name.split('/')[-1].split(".")[0]
                with open(name + ".json", 'w') as f:
                    json.dump(d_model.config, f)
                torch.save(d_model.state_dict(), d_file_name)
            except:
                torch.save(d_model, d_file_name)


def train_wgan(dataloader, g_model: nn.Module, g_opt: torch.optim, d_model: nn.Module, d_opt: torch.optim,
               loss_func: nn.Module = None, Epoch: int = 100,
               g_file_name: str = None, d_file_name: str = None, device: str = "cuda",
               clip_grad: bool = False, in_dim: int = None, g_model_train_step: int = 5, lambda_gp: int = 10,
               create_graph: bool = True, retain_graph: bool = True):
    if loss_func is None:
        loss_func = WGAN_GP_DLoss(lambda_gp=lambda_gp)
    if in_dim is None:
        if hasattr(g_model, 'in_dim'):
            in_dim = g_model.in_dim
        else:
            raise ValueError("生成器模型必须定义 'in_dim' 属性或显式传入 in_dim 参数")
    scaler = GradScaler()
    for epoch in range(Epoch):
        g_model.train()
        d_model.train()
        for index, (tx, *_) in enumerate(tqdm(dataloader, desc=f"{epoch}/{Epoch}")):
            tx = tx.to(device)
            norise = torch.randn(size=(tx.shape[0], in_dim)).to(device)
            fake_image = g_model(norise)
            d_opt.zero_grad()
            with autocast():
                d_loss = loss_func(model=d_model, real_samples=tx, fake_samples=fake_image.detach(),
                                   create_graph=create_graph,
                                   retain_graph=retain_graph)
                if torch.isnan(d_loss).any():
                    print("NaN detected! Skipping step")
                    d_opt.zero_grad()
                    continue
            scaler.scale(d_loss).backward()  # 缩放损失并反向传播
            if clip_grad:
                scaler.unscale_(d_opt)
                torch.nn.utils.clip_grad_norm_(d_model.parameters(), max_norm=1.0)
            scaler.step(d_opt)  # 更新参数
            scaler.update()
            if index % g_model_train_step == 0:
                g_opt.zero_grad()
                with autocast():
                    g_loss = -d_model(fake_image).mean()
                    if torch.isnan(g_loss).any():
                        print("NaN detected! Skipping step")
                        g_opt.zero_grad()
                        continue
                scaler.scale(g_loss).backward()  # 缩放损失并反向传播
                if clip_grad:
                    scaler.unscale_(g_opt)
                    torch.nn.utils.clip_grad_norm_(g_model.parameters(), max_norm=1.0)
                scaler.step(g_opt)  # 更新参数
                scaler.update()
        if g_file_name:
            try:
                name = g_file_name.split('/')[-1].split(".")[0]
                with open(name + ".json", 'w') as f:
                    json.dump(g_model.config, f)
                torch.save(g_model.state_dict(), g_file_name)
            except:
                torch.save(g_model, g_file_name)
        if d_file_name:
            try:
                name = d_file_name.split('/')[-1].split(".")[0]
                with open(name + ".json", 'w') as f:
                    json.dump(d_model.config, f)
                torch.save(d_model.state_dict(), d_file_name)
            except:
                torch.save(d_model, d_file_name)


def train_AutoEncoderKl(dataloader, model: nn.Module, vae_opt: torch.optim = None, dis_opt: torch.optim = None,
                        Epoch: int = 100,
                        file_name: str = None,
                        device: str = "cuda",
                        clip_grad: bool = False, use_gan: bool = False, gan_model_train_step: int = 5):
    scaler = GradScaler()
    if vae_opt is None:
        vae_opt, dis_opt = model.set_optim(lr1=1e-4, lr2=1e-4)
    if use_gan is False:
        gan_model_train_step = 1
    for epoch in range(Epoch):
        model.train()
        for index, (tx, *_) in enumerate(tqdm(dataloader, desc=f"{epoch}/{Epoch}")):
            tx = tx.to(device)
            if use_gan:
                dis_opt.zero_grad()
                with autocast():
                    loss = model.training_step(x=tx, mode="dis", use_gan=use_gan)
                    if torch.isnan(loss).any():
                        print("NaN detected! Skipping step")
                        dis_opt.zero_grad()
                        continue
                scaler.scale(loss).backward()  # 缩放损失并反向传播
                if clip_grad:
                    scaler.unscale_(dis_opt)
                    torch.nn.utils.clip_grad_norm_(model.loss.parameters(), max_norm=1.0)
                scaler.step(dis_opt)  # 更新参数
                scaler.update()
            if index % gan_model_train_step == 0:
                vae_opt.zero_grad()
                with autocast():
                    loss = model.training_step(x=tx, mode="vae", use_gan=use_gan)
                    if torch.isnan(loss).any():
                        print("NaN detected! Skipping step")
                        vae_opt.zero_grad()
                        continue
                scaler.scale(loss).backward()  # 缩放损失并反向传播
                if clip_grad:
                    scaler.unscale_(vae_opt)
                    torch.nn.utils.clip_grad_norm_(model.vae.parameters(), max_norm=1.0)
                scaler.step(vae_opt)  # 更新参数
                scaler.update()
        if file_name:
            try:
                name = file_name.split('/')[-1].split(".")[0]
                with open(name + ".json", 'w') as f:
                    json.dump(model.config, f)
                torch.save(model.state_dict(), file_name)
            except:
                torch.save(model, file_name)
