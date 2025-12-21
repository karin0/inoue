while = {cond, body ↦
  *cond ? *body *while
};

for = {init, cond, step, body ↦
  *init ;
  _body = $body;
  body = { ↦ *_body *step };
  *while
};

n = $0;

{ init ↦ i = "0" };
{ cond ↦ "i<n" };
{ step ↦ i = "i+1" };
{ body ↦ a[i] = "randint(10000)" };
*for;

flag = 1;
{ cond ↦ "i<n-1" ? flag : 0 };
body = {@; ↦
  t = 0; flag ^ t;
  { init ↦ j = "0" };
  { cond ↦ "j < n-i-1" };
  { step ↦ j = "j+1" };
  { body ↦
    "a[j]>a[j+1]" ? a[j] ^ a["j+1"]; t = 1 :
  };
  *for;
  flag ^ t;
};
*for;

{ cond ↦ "i<n" };
{ body ↦ a[i]; '\n' };
*for;

{ cond ↦ "i<n-1" };
{ body ↦ "a[i]>a[i+1]" ? $ERR };
*for;
