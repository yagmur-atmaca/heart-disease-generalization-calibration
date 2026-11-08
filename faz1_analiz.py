# -*- coding: utf-8 -*-
# Uygulamalı Yapay Zekâ Faz 1 | Yağmur Atmaca
# Veri keşfi, kaynak örtüşmesi ve sızıntısız ön işleme. Modeller Faz 2'de eğitilecektir.
# %%
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

ROOT=Path.cwd()
if not (ROOT/"veri"/"processed.cleveland.data").exists():
    ROOT=ROOT.parent
OUT=ROOT/"ciktilar";FIG=ROOT/"grafikler"
OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
GIRDI=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
COL=GIRDI+["num"]
NUM=["age","trestbps","chol","thalach","oldpeak"]
CAT=["sex","cp","fbs","restecg","exang"]
DOSYA={"Cleveland":"processed.cleveland.data","Statlog":"heart.dat","Hungarian":"processed.hungarian.data"}
SHA={"Cleveland":"97e936c65918c6cda5c1c62661c548ed69afc7b8","Statlog":"5d17e1692c1eb76ce07da3588852679f6f921a43","Hungarian":"23b36897a9e72ce2f2fd35edc455630828ca15c8"}

# %% 1 — Dosya bütünlüğü ve etiketler
ham={};data={};metadata=[]
for name,filename in DOSYA.items():
    p=ROOT/"veri"/filename
    raw=p.read_bytes()
    gitsha=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
    assert gitsha==SHA[name],f"Kaynak dosya değişmiş: {name}"
    sep=r"\s+" if name=="Statlog" else ","
    df=pd.read_csv(p,names=COL,header=None,sep=sep,na_values="?")
    assert len(df)=={"Cleveland":303,"Statlog":270,"Hungarian":294}[name]
    df["hedef"]=(df.num.eq(2) if name=="Statlog" else df.num.gt(0)).astype(int)
    ham[name]=df
    data[name]=df.drop_duplicates(subset=COL).reset_index(drop=True)
    metadata.append({"Küme":name,"SHA256":hashlib.sha256(raw).hexdigest(),"GitBlobSHA1":gitsha})
pd.DataFrame(metadata).to_csv(OUT/"sha256_veri_kaynaklari.csv",index=False)

# %% 2 — Sayılar, sınıflar ve eksiklik
stats=[];missing=[];classes=[]
for name,df in data.items():
    m=df[GIRDI].isna().sum()
    count=df.hedef.value_counts().reindex([0,1],fill_value=0)
    stats.append({"Küme":name,"Ham":len(ham[name]),"Tekil":len(df),"Mükerrer":len(ham[name])-len(df),
                  "Sınıf0":int(count[0]),"Sınıf1":int(count[1]),"Eksik":int(m.sum()),
                  "Yaş ortalaması":round(df.age.mean(),2)})
    for c in [0,1]:classes.append({"Küme":name,"Sınıf":str(c),"Adet":int(count[c])})
    for col in GIRDI:missing.append({"Küme":name,"Alan":col,"Eksik":int(m[col])})
stats=pd.DataFrame(stats);missing=pd.DataFrame(missing);classes=pd.DataFrame(classes)
assert list(stats.Tekil)==[303,270,293] and list(stats.Eksik)==[6,0,778]
assert int(data["Hungarian"][NUM+CAT].isna().sum().sum())==34
stats.to_csv(OUT/"veri_ozeti.csv",index=False)
missing.to_csv(OUT/"eksik_hucreler.csv",index=False)
classes.to_csv(OUT/"sinif_sayilari.csv",index=False)
print(stats.to_string(index=False))

# %% 3 — Tam kayıt örtüşmesi (hasta kimliği değildir)
def keys(df):
    return set(df[GIRDI+["hedef"]].apply(
        lambda row:"|".join("?" if pd.isna(x) else f"{float(x):.6f}" for x in row),axis=1))
sets={name:keys(df) for name,df in data.items()}
overlap=[{"Kaynak":a,"Hedef":b,"Tam eşleşme":len(sets[a]&sets[b])}
         for a,b in [("Cleveland","Statlog"),("Cleveland","Hungarian"),("Statlog","Hungarian")]]
assert overlap[0]["Tam eşleşme"]==270 and overlap[1]["Tam eşleşme"]==0
assert len(set().union(*sets.values()))==596
pd.DataFrame(overlap).to_csv(OUT/"kume_ortusmesi.csv",index=False)
print(pd.DataFrame(overlap).to_string(index=False))
print("866 toplam kaynak satırı; 596 farklı tam kayıt örüntüsü. Benzersiz hasta sayısı değildir.")

# %% 4 — Aykırı değer ve korelasyon
c=data["Cleveland"]
flags=pd.DataFrame(False,index=c.index,columns=NUM)
outlier=[]
for a in NUM:
    q1,q3=c[a].quantile([.25,.75]);iqr=q3-q1
    lo=q1-1.5*iqr; hi=q3+1.5*iqr
    flags[a]=(c[a]<lo)|(c[a]>hi)
    outlier.append({"Alan":a,"Q1":q1,"Medyan":c[a].median(),"Q3":q3,"IQR":iqr,"Alt":lo,"Üst":hi,"İşaretleme":int(flags[a].sum())})
assert flags.to_numpy().sum()==20 and flags.any(axis=1).sum()==19
pd.DataFrame(outlier).to_csv(OUT/"aykiri_ozet.csv",index=False)
correlation=c[NUM].corr()
correlation.to_csv(OUT/"pearson_sayisal.csv")

# %% 5 — Gerçek görseller
sns.set_theme(style="whitegrid")
fig,ax=plt.subplots(figsize=(8,4))
sns.barplot(data=classes,x="Küme",y="Adet",hue="Sınıf",ax=ax)
ax.set(title="Kaynağa göre ikili sınıf sayıları");fig.tight_layout()
fig.savefig(FIG/"01_sinif_dagilimi.png");plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5))
sns.heatmap(missing.pivot(index="Alan",columns="Küme",values="Eksik"),
            annot=True,fmt="d",cmap="YlOrRd",ax=ax)
ax.set(title="Ham 13 girdide eksik hücreler");fig.tight_layout()
fig.savefig(FIG/"02_eksik_veri_isi_haritasi.png");plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4))
for name,df in data.items():
    ax.hist(df.age,alpha=.45,label=name,bins=13)
ax.set(title="Veri kümelerinde yaş dağılımı",xlabel="Yaş");ax.legend();fig.tight_layout()
fig.savefig(FIG/"03_yas_yogunlugu.png");plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(10,4))
for ax,col in zip(axs,["trestbps","chol","oldpeak"]):
    sns.boxplot(data=c,y=col,ax=ax)
fig.suptitle("Cleveland: gerçek boxplot'lar");fig.tight_layout()
fig.savefig(FIG/"04_boxplot.png");plt.close(fig)
fig,ax=plt.subplots(figsize=(7,5))
sns.heatmap(correlation,annot=True,vmin=-1,vmax=1,cmap="vlag",ax=ax)
ax.set(title="Cleveland sürekli değişkenler Pearson r");fig.tight_layout()
fig.savefig(FIG/"05_pearson_isiharitasi.png");plt.close(fig)
fig,ax=plt.subplots(figsize=(7,3))
ax.bar(["C-S","C-H","S-H"],[o["Tam eşleşme"] for o in overlap])
ax.set(ylabel="Tam kayıt eşleşmesi",title="13 özellik ve hedef bazında örtüşme")
fig.tight_layout();fig.savefig(FIG/"06_kayit_ortusmesi.png");plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
sns.countplot(data=c,x="cp",hue="hedef",ax=ax)
ax.set(xlabel="Göğüs ağrısı türü",ylabel="Kayıt sayısı",title="Cleveland: göğüs ağrısı ve etiket")
fig.tight_layout();fig.savefig(FIG/"07_gogus_agrisi.png");plt.close(fig)
print("Yedi gerçek grafik grafikler/ klasörüne kaydedildi.")

# %% 6 — Sadece eğitim katında fit; holdout ve Hungarian'da transform
def preprocessor(categorical):
    num=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",RobustScaler())])
    cat=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                  ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
    return ColumnTransformer([("num",num,NUM),("cat",cat,categorical)],remainder="drop")

X=c[GIRDI];y=c.hedef
train,test=next(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(X,y))
full=preprocessor(["sex","cp","fbs","restecg","exang","slope","ca","thal"])
A=full.fit_transform(X.iloc[train])
B=full.transform(X.iloc[test])
common=preprocessor(CAT)
C=common.fit_transform(X.iloc[train][NUM+CAT])
D=common.transform(data["Hungarian"][NUM+CAT])
assert (A.shape,B.shape,C.shape,D.shape)==((242,28),(61,28),(242,18),(293,18))
assert np.isfinite(A).all() and np.isfinite(B).all() and np.isfinite(C).all() and np.isfinite(D).all()
assert set(train).isdisjoint(set(test))
print("Tam matris:",A.shape,B.shape,"Ortak 10 girdi:",C.shape,D.shape)

# %% 7 — Son kayıt: model skorları Faz 2 kapsamındadır
summary={"ham_satir_toplami":866,"farkli_tam_kayit":596,"hungarian_eksik":778,
         "hungarian_ortak_10_eksik":34,"tam_egitim_matrisi":list(A.shape),
         "ortak_egitim_matrisi":list(C.shape),"hungarian_donusum_matrisi":list(D.shape),
         "aykiri_isaretleme":20,"aykiri_farkli_kayit":19,
         "model_egitimi":"Faz 2 - uygulanmadı",
         "nested_cv":"Faz 2 - uygulanmadı",
         "Brier_skora_ayrisimi":"Reliability - Resolution + Uncertainty"}
(OUT/"faz1_dogrulama_ozeti.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print("Faz 1 tamamlandı: EDA ve ön işleme. Model başarısı üretilmedi.")
