local M = {}

M.btrunc = {{epoch =    1, len =  1, load_len =  -1, lr = 5e-4},
            {epoch =  601, len =  5, load_len =  -5, lr = 5e-4},
            {epoch = 2000, len =  5, load_len =  -5, lr = 5e-5},
            {epoch = 3000, len =  5, load_len =  -5, lr = 5e-6},
            {epoch = 4001, len = 10, load_len = -10, lr = 5e-5},
            {epoch = 7000, len = 10, load_len = -10, lr = 1e-5},
            {epoch = 9000, len = 21, load_len = 21, lr = 1e-6}}

M.ac     = {{epoch =  601, len =  5, load_len =  -5, lr = 5e-4, kld = 1e-4},
            {epoch = 5000, len =  5, load_len =  -5, lr = 5e-5, kld = 1e-4},
            {epoch = 7000, len = 10, load_len = -10, lr = 5e-5, kld = 1e-4},
            {epoch = 8000, len = 10, load_len = -10, lr = 1e-5, kld = 1e-5},
            {epoch = 9000, len = 15, load_len =  15, lr = 1e-6, kld = 1e-5}}

return M
