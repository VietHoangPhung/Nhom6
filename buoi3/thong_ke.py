import pandas as pd
from numpy import array
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv('diemPython.csv',index_col=0,header = 0)
in_data = array(df.iloc[:,:])
print(in_data)
print('Tong so sinh vien di thi :')
tongsv= in_data[:,1]
print(np.sum(tongsv))
diemA = in_data[:,3]
diemBc = in_data[:,4]
print('Tong sv:',tongsv)
maxa = diemA.max()
i = np.where(diemA == maxa)
print('lop co nhieu diem A la {0} co {1} sv dat diem A'.format(in_data[i,0],maxa))




def diemcaonhatcuatunglop( a):
    caclop=in_data[a,2:11]
    print(caclop)
    maxb = caclop.max()
    maxbb= maxb
    idd= np.where(caclop == maxb)
    if maxbb==in_data[a,2]:
        aa="Loai A+"
        print('lop co diem cao nhat la diem {0} va co {1} sinh vien dat duoc'.format(aa, maxb))
    elif maxbb==in_data[a,3]:
        b="Loai A"
        print('lop co diem cao nhat la diem {0} va co {1} sinh vien dat duoc'.format(b, maxb))
    elif maxbb==in_data[a,4]:
        c="Loai B+"
        print('lop co diem cao nhat la diem {0} va co {1} sinh vien dat duoc'.format(c, maxb))
    elif maxbb==in_data[a,5]:
        d="Loai B"
        print('lop co diem cao nhat la diem {0} va co {1} sinh vien dat duoc'.format(d, maxb))
    elif maxbb==in_data[a,6]:
        e="Loai C+"
        print('lop co diem cao nhat la diem {0} va co {1} sinh vien dat duoc'.format(e, maxb))
    elif maxbb==in_data[a,7]:
        f="Loai C"
        print('lop co diem cao nhat la diem {0} va co {1} sinh vien dat duoc'.format(f, maxb))
    elif maxbb==in_data[a,8]:
        g="Loai D+"
        print('lop co diem cao nhat la diem {0} va co {1} sinh vien dat duoc'.format(g, maxb))
    elif maxbb==in_data[a,9]:
        h="Loai D"
        print('lop co diem cao nhat la diem {0} va co {1} sinh vien dat duoc'.format(h, maxb))
    elif maxbb==in_data[a,10]:
        j="Loai F"
        print('lop co diem cao nhat la diem {0} va co {1} sinh vien dat duoc'.format(j, maxb))


chonlop=int(input("nhap stt lop muon chon: "))
diemcaonhatcuatunglop(chonlop)

def diemthapnhatcuatunglop( a):
    caclop=in_data[a,2:11]
    print(caclop)
    maxb = caclop.min()
    maxbb= maxb
    idd= np.where(caclop == maxb)
    if maxbb==in_data[a,2]:
        aa="Loai A+"
        print('lop co diem thap nhat la diem {0} va co {1} sinh vien dat duoc'.format(aa, maxb))
    elif maxbb==in_data[a,3]:
        b="Loai A"
        print('lop co diem thap nhat la diem {0} va co {1} sinh vien dat duoc'.format(b, maxb))
    elif maxbb==in_data[a,4]:
        c="Loai B+"
        print('lop co diem thap nhat la diem {0} va co {1} sinh vien dat duoc'.format(c, maxb))
    elif maxbb==in_data[a,5]:
        d="Loai B"
        print('lop co diem thap nhat la diem {0} va co {1} sinh vien dat duoc'.format(d, maxb))
    elif maxbb==in_data[a,6]:
        e="Loai C+"
        print('lop co diem thap nhat la diem {0} va co {1} sinh vien dat duoc'.format(e, maxb))
    elif maxbb==in_data[a,7]:
        f="Loai C"
        print('lop co diem thap nhat la diem {0} va co {1} sinh vien dat duoc'.format(f, maxb))
    elif maxbb==in_data[a,8]:
        g="Loai D+"
        print('lop co diem thap nhat la diem {0} va co {1} sinh vien dat duoc'.format(g, maxb))
    elif maxbb==in_data[a,9]:
        h="Loai D"
        print('lop co diem thap nhat la diem {0} va co {1} sinh vien dat duoc'.format(h, maxb))
    elif maxbb==in_data[a,10]:
        j="Loai F"
        print('lop co diem thap nhat la diem {0} va co {1} sinh vien dat duoc'.format(j, maxb))



chonlopthapnhat=int(input("nhap stt lop muon chon: "))
diemthapnhatcuatunglop(chonlopthapnhat)  

