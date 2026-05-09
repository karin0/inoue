{for ↦
  *cond ? *body *step *for
};

n = $0;

i = 0;
{ cond ↦ i < n }
{ step ↦ ++i }
{ body ↦ a[i] = randint(10000) }
*for;

qs = { @; l, r ↦
  l < r ? {
    pivot = a[r]
    p = l-1

    j = $l
    { cond ↦ j < r }
    { step ↦ ++j }
    { body ↦ a[j] < pivot ? a[++p] ^ a[j] }
    *for

    a[++p] ^ a[r]

    qs(l, p-1)
    qs(p+1, r)
  }
};

// .l = 0; .r = n-1; *qs;
qs(0, n-1);

i = 0;
{ body ↦ a[i]; '\n' }
*for;

i = 0;
{ cond ↦ i<n-1 }
{ body ↦ a[i]>a[i+1] ? $ERR }
*for;
