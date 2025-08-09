set number
set tabstop=4
set softtabstop=4
set shiftwidth=4
" set textwidth=87
set colorcolumn=88
set expandtab
set autoindent
set fileformat=unix

let python_highlight_all=1
syntax on

highlight ColorColumn ctermbg=16                                                   
highlight DiffAdd    cterm=none ctermfg=223 ctermbg=23 gui=none guifg=bg guibg=#e1ddbf
highlight DiffDelete cterm=none ctermfg=223 ctermbg=17 gui=none guifg=bg guibg=#e1ddbf
highlight DiffChange cterm=none ctermfg=223 ctermbg=18 gui=none guifg=bg guibg=#e1ddbf
highlight DiffText   cterm=none ctermfg=223 ctermbg=24 gui=none guifg=bg guibg=#e1ddbf
" 23  Navy blue                                                                    
" 66  Turquoisey green                                                             
" 223 Off white                                                                    
" Inspired by                                                                      
" Retro Rocks palette from SlideStream                                             
" https://www.slideteam.net/blog/9-beautiful-color-palettes-for-designing-powerful-powerpoint-slides/
" #04253a : dark blue                                                              
" #e1ddbf : tan                                                                    
" #4c837a : green                                                                  
                                                                                   
autocmd BufRead,BufNewFile *.htm,*.html setlocal tabstop=2 shiftwidth=2 softtabstop=2
