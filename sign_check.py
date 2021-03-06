import hashlib
from Crypto import Random
from Crypto.PublicKey import RSA
from Crypto.Hash import SHA
from Crypto.Signature import PKCS1_v1_5 as Sig_pk

def check(label_num,sign,paddings):
    bits=-1
    while label_num>0:
        label_num=int(label_num/2)
        bits+=1
    if len(sign)%bits !=0:
        sign+='0'*(bits-len(sign)%bits)

    check_list=[]
    for i in range(int(len(sign)/bits)):
        check_list.append(int(sign[i*bits:(i+1)*bits],2))

    return check_list,check_repeat(check_list*2,paddings)

def check_repeat(s,length):
    if len(s)<=length:
        return True
    left=0

    while left<=len(s)-length-1:
        right = left+1
        while right<=len(s)-length and right-left<len(s)/2:
            if s[left]==s[right] :
                for i in range(length):
                    if s[left+i]!=s[right+i]:
                        break
                    if i==length-1:
                        return False
            right+=1
        left+=1
    return True

def key_gen():
    random_gen = Random.new().read

    # 生成秘钥对实例对象：1024是秘钥的长度
    rsa = RSA.generate(1024, random_gen)

    # 获取公钥，保存到文件
    private_pem = rsa.exportKey()
    with open('private.pem', 'wb') as f:
        f.write(private_pem)

    # 获取私钥保存到文件
    public_pem = rsa.publickey().exportKey()
    with open('public.pem', 'wb') as f:
        f.write(public_pem)

def sign_gen(name,key_path):# 待签名内容

    # 获取私钥
    key = open(key_path, 'r').read()
    rsakey = RSA.importKey(key)
    # 根据sha算法处理签名内容  (此处的hash算法不一定是sha,看开发)
    data = SHA.new(name.encode())
    # 私钥进行签名
    sig_pk = Sig_pk.new(rsakey)
    sign = sig_pk.sign(data)
    return sign
def sign_verfy(origin,sign,key_path):
    key = open(key_path).read()
    rsakey = RSA.importKey(key)
    signer = Sig_pk.new(rsakey)
    sha_name = SHA.new(origin.encode())
    result = signer.verify(sha_name, sign)
    return result

def binencode(sign):
    return (''.join(['{:0>4b}'.format(int(c,16)) for c in sign.hex()]))


def get_labels(text,types,paddings):
    flag=False
    labels=[]
    while not flag:
        key_gen()
        sign_bin = binencode(sign_gen("123", 'private.pem'))
        labels, flag = check(types, sign_bin, paddings)
    a = sign_verfy(text, sign_gen(text, 'private.pem'), 'public.pem')
    print(a)

    return labels

print(get_labels("123",10,5))
