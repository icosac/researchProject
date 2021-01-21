#include<trig.cuh>

#define __HI(x) *(1+(int*)&x)
#define __LO(x) *(int*)&x
#define __HIp(x) *(1+(int*)x)
#define __LOp(x) *(int*)x


BOTH double __kernel_sin(double x, double y, int iy)
{
	double z,r,v;
	int ix;
	ix = __HI(x)&0x7fffffff;	/* high word of x */
	if(ix<0x3e400000)			/* |x| < 2**-27 */
	   {if((int)x==0) return x;}		/* generate inexact */
	z	=  x*x;
	v	=  z*x;
	r	=  S2+z*(S3+z*(S4+z*(S5+z*S6)));
	if(iy==0) return x+v*(S1+z*r);
	else      return x-((z*(half*y-v*r)-y)-v*S1);
}


BOTH double __kernel_cos(double x, double y)
{
	double a,hz,z,r,qx;
	int ix;
	ix = __HI(x)&0x7fffffff;	/* ix = |x|'s high word*/
	if(ix<0x3e400000) {			/* if x < 2**27 */
	    if(((int)x)==0) return one;		/* generate inexact */
	}
	z  = x*x;
	r  = z*(C1+z*(C2+z*(C3+z*(C4+z*(C5+z*C6)))));
	if(ix < 0x3FD33333) 			/* if |x| < 0.3 */ 
	    return one - (0.5*z - (z*r - x*y));
	else {
	    if(ix > 0x3fe90000) {		/* x > 0.78125 */
		qx = 0.28125;
	    } else {
	        __HI(qx) = ix-0x00200000;	/* x/4 */
	        __LO(qx) = 0;
	    }
	    hz = 0.5*z-qx;
	    a  = one-qx;
	    return a - (hz - (z*r-x*y));
	}
}


BOTH int __kernel_rem_pio2(double *x, double *y, int e0, int nx, int prec, const int *ipio2) 
{
	int jz,jx,jv,jp,jk,carry,n,iq[20],i,j,k,m,q0,ih;
	double z,fw,f[20],fq[20],q[20];

    /* initialize jk*/
	jk = init_jk[prec];
	jp = jk;

    /* determine jx,jv,q0, note that 3>q0 */
	jx =  nx-1;
	jv = (e0-3)/24; if(jv<0) jv=0;
	q0 =  e0-24*(jv+1);

    /* set up f[0] to f[jx+jk] where f[jx+jk] = ipio2[jv+jk] */
	j = jv-jx; m = jx+jk;
	for(i=0;i<=m;i++,j++) f[i] = (j<0)? zero : (double) ipio2[j];

    /* compute q[0],q[1],...q[jk] */
	for (i=0;i<=jk;i++) {
	    for(j=0,fw=0.0;j<=jx;j++) fw += x[j]*f[jx+i-j]; q[i] = fw;
	}

	jz = jk;
recompute:
    /* distill q[] into iq[] reversingly */
	for(i=0,j=jz,z=q[jz];j>0;i++,j--) {
	    fw    =  (double)((int)(twon24* z));
	    iq[i] =  (int)(z-two24*fw);
	    z     =  q[j-1]+fw;
	}

    /* compute n */
	z  = scalbn(z,q0);		/* actual value of z */
	z -= 8.0*floor(z*0.125);		/* trim off integer >= 8 */
	n  = (int) z;
	z -= (double)n;
	ih = 0;
	if(q0>0) {	/* need iq[jz-1] to determine n */
	    i  = (iq[jz-1]>>(24-q0)); n += i;
	    iq[jz-1] -= i<<(24-q0);
	    ih = iq[jz-1]>>(23-q0);
	} 
	else if(q0==0) ih = iq[jz-1]>>23;
	else if(z>=0.5) ih=2;

	if(ih>0) {	/* q > 0.5 */
	    n += 1; carry = 0;
	    for(i=0;i<jz ;i++) {	/* compute 1-q */
		j = iq[i];
		if(carry==0) {
		    if(j!=0) {
			carry = 1; iq[i] = 0x1000000- j;
		    }
		} else  iq[i] = 0xffffff - j;
	    }
	    if(q0>0) {		/* rare case: chance is 1 in 12 */
	        switch(q0) {
	        case 1:
	    	   iq[jz-1] &= 0x7fffff; break;
	    	case 2:
	    	   iq[jz-1] &= 0x3fffff; break;
	        }
	    }
	    if(ih==2) {
		z = one - z;
		if(carry!=0) z -= scalbn(one,q0);
	    }
	}

    /* check if recomputation is needed */
	if(z==zero) {
	    j = 0;
	    for (i=jz-1;i>=jk;i--) j |= iq[i];
	    if(j==0) { /* need recomputation */
		for(k=1;iq[jk-k]==0;k++);   /* k = no. of terms needed */

		for(i=jz+1;i<=jz+k;i++) {   /* add q[jz+1] to q[jz+k] */
		    f[jx+i] = (double) ipio2[jv+i];
		    for(j=0,fw=0.0;j<=jx;j++) fw += x[j]*f[jx+i-j];
		    q[i] = fw;
		}
		jz += k;
		goto recompute;
	    }
	}

  /* chop off zero terms */
	if(z==0.0) {
	    jz -= 1; q0 -= 24;
	    while(iq[jz]==0) { jz--; q0-=24;}
	} else { /* break z into 24-bit if necessary */
	    z = scalbn(z,-q0);
	    if(z>=two24) { 
		fw = (double)((int)(twon24*z));
		iq[jz] = (int)(z-two24*fw);
		jz += 1; q0 += 24;
		iq[jz] = (int) fw;
	    } else iq[jz] = (int) z ;
	}

    /* convert integer "bit" chunk to floating-point value */
	fw = scalbn(one,q0);
	for(i=jz;i>=0;i--) {
	    q[i] = fw*(double)iq[i]; fw*=twon24;
	}

    /* compute PIo2[0,...,jp]*q[jz,...,0] */
	for(i=jz;i>=0;i--) {
	    for(fw=0.0,k=0;k<=jp&&k<=jz-i;k++) fw += PIo2[k]*q[i+k];
	    fq[jz-i] = fw;
	}

    /* compress fq[] into y[] */
	switch(prec) {
	    case 0:
		fw = 0.0;
		for (i=jz;i>=0;i--) fw += fq[i];
		y[0] = (ih==0)? fw: -fw; 
		break;
	    case 1:
	    case 2:
		fw = 0.0;
		for (i=jz;i>=0;i--) fw += fq[i]; 
		y[0] = (ih==0)? fw: -fw; 
		fw = fq[0]-fw;
		for (i=1;i<=jz;i++) fw += fq[i];
		y[1] = (ih==0)? fw: -fw; 
		break;
	    case 3:	/* painful */
		for (i=jz;i>0;i--) {
		    fw      = fq[i-1]+fq[i]; 
		    fq[i]  += fq[i-1]-fw;
		    fq[i-1] = fw;
		}
		for (i=jz;i>1;i--) {
		    fw      = fq[i-1]+fq[i]; 
		    fq[i]  += fq[i-1]-fw;
		    fq[i-1] = fw;
		}
		for (fw=0.0,i=jz;i>=2;i--) fw += fq[i]; 
		if(ih==0) {
		    y[0] =  fq[0]; y[1] =  fq[1]; y[2] =  fw;
		} else {
		    y[0] = -fq[0]; y[1] = -fq[1]; y[2] = -fw;
		}
	}
	return n&7;
}


BOTH int __ieee754_rem_pio2(double x, double *y)
{
	double z,w,t,r,fn;
	double tx[3];
	int e0,i,j,nx,n,ix,hx;

	hx = __HI(x);		/* high word of x */
	ix = hx&0x7fffffff;
	if(ix<=0x3fe921fb)   /* |x| ~<= pi/4 , no need for reduction */
	    {y[0] = x; y[1] = 0; return 0;}
	if(ix<0x4002d97c) {  /* |x| < 3pi/4, special case with n=+-1 */
	    if(hx>0) { 
		z = x - pio2_1;
		if(ix!=0x3ff921fb) { 	/* 33+53 bit pi is good enough */
		    y[0] = z - pio2_1t;
		    y[1] = (z-y[0])-pio2_1t;
		} else {		/* near pi/2, use 33+33+53 bit pi */
		    z -= pio2_2;
		    y[0] = z - pio2_2t;
		    y[1] = (z-y[0])-pio2_2t;
		}
		return 1;
	    } else {	/* negative x */
		z = x + pio2_1;
		if(ix!=0x3ff921fb) { 	/* 33+53 bit pi is good enough */
		    y[0] = z + pio2_1t;
		    y[1] = (z-y[0])+pio2_1t;
		} else {		/* near pi/2, use 33+33+53 bit pi */
		    z += pio2_2;
		    y[0] = z + pio2_2t;
		    y[1] = (z-y[0])+pio2_2t;
		}
		return -1;
	    }
	}
	if(ix<=0x413921fb) { /* |x| ~<= 2^19*(pi/2), medium size */
	    t  = fabs(x);
	    n  = (int) (t*invpio2+half);
	    fn = (double)n;
	    r  = t-fn*pio2_1;
	    w  = fn*pio2_1t;	/* 1st round good to 85 bit */
	    if(n<32&&ix!=npio2_hw[n-1]) {	
		y[0] = r-w;	/* quick check no cancellation */
	    } else {
	        j  = ix>>20;
	        y[0] = r-w; 
	        i = j-(((__HI(y[0]))>>20)&0x7ff);
	        if(i>16) {  /* 2nd iteration needed, good to 118 */
		    t  = r;
		    w  = fn*pio2_2;	
		    r  = t-w;
		    w  = fn*pio2_2t-((t-r)-w);	
		    y[0] = r-w;
		    i = j-(((__HI(y[0]))>>20)&0x7ff);
		    if(i>49)  {	/* 3rd iteration need, 151 bits acc */
		    	t  = r;	/* will cover all possible cases */
		    	w  = fn*pio2_3;	
		    	r  = t-w;
		    	w  = fn*pio2_3t-((t-r)-w);	
		    	y[0] = r-w;
		    }
		}
	    }
	    y[1] = (r-y[0])-w;
	    if(hx<0) 	{y[0] = -y[0]; y[1] = -y[1]; return -n;}
	    else	 return n;
	}
    /* 
     * all other (large) arguments
     */
	if(ix>=0x7ff00000) {		/* x is inf or NaN */
	    y[0]=y[1]=x-x; return 0;
	}
    /* set z = scalbn(|x|,ilogb(x)-23) */
	__LO(z) = __LO(x);
	e0 	= (ix>>20)-1046;	/* e0 = ilogb(z)-23; */
	__HI(z) = ix - (e0<<20);
	for(i=0;i<2;i++) {
		tx[i] = (double)((int)(z));
		z     = (z-tx[i])*two24;
	}
	tx[2] = z;
	nx = 3;
	while(tx[nx-1]==zero) nx--;	/* skip zero term */
	n  =  __kernel_rem_pio2(tx,y,e0,nx,2,two_over_pi);
	if(hx<0) {y[0] = -y[0]; y[1] = -y[1]; return -n;}
	return n;
}


BOTH double my_sin(double x) {
  double y[2], z=0.0;
	int n, ix;

    /* High word of x. */
	ix = __HI(x);

    /* |x| ~< pi/4 */
	ix &= 0x7fffffff;
	if(ix <= 0x3fe921fb) return __kernel_sin(x,z,0);

    /* sin(Inf or NaN) is NaN */
	else if (ix>=0x7ff00000) return x-x;

    /* argument reduction needed */
	else {
	    n = __ieee754_rem_pio2(x,y);
	    switch(n&3) {
		case 0: return  __kernel_sin(y[0],y[1],1);
		case 1: return  __kernel_cos(y[0],y[1]);
		case 2: return -__kernel_sin(y[0],y[1],1);
		default:
			return -__kernel_cos(y[0],y[1]);
	    }
	}
}


BOTH double my_cos(double x) {
  double y[2],z=0.0;
	int n, ix;

    /* High word of x. */
	ix = __HI(x);

    /* |x| ~< pi/4 */
	ix &= 0x7fffffff;
	if(ix <= 0x3fe921fb) return __kernel_cos(x,z);

    /* cos(Inf or NaN) is NaN */
	else if (ix>=0x7ff00000) return x-x;

    /* argument reduction needed */
	else {
	    n = __ieee754_rem_pio2(x,y);
	    switch(n&3) {
		case 0: return  __kernel_cos(y[0],y[1]);
		case 1: return -__kernel_sin(y[0],y[1],1);
		case 2: return -__kernel_cos(y[0],y[1]);
		default:
		        return  __kernel_sin(y[0],y[1],1);
	    }
	}
}


BOTH double my_atan(double x)
{
	double w,s1,s2,z;
	int ix,hx,id;

	hx = __HI(x);
	ix = hx&0x7fffffff;
	if(ix>=0x44100000) {	/* if |x| >= 2^66 */
	    if(ix>0x7ff00000||
		(ix==0x7ff00000&&(__LO(x)!=0)))
		return x+x;		/* NaN */
	    if(hx>0) return  atanhi[3]+atanlo[3];
	    else     return -atanhi[3]-atanlo[3];
	} if (ix < 0x3fdc0000) {	/* |x| < 0.4375 */
	    if (ix < 0x3e200000) {	/* |x| < 2^-29 */
		if(huge+x>one) return x;	/* raise inexact */
	    }
	    id = -1;
	} else {
	x = fabs(x);
	if (ix < 0x3ff30000) {		/* |x| < 1.1875 */
	    if (ix < 0x3fe60000) {	/* 7/16 <=|x|<11/16 */
		id = 0; x = (2.0*x-one)/(2.0+x); 
	    } else {			/* 11/16<=|x|< 19/16 */
		id = 1; x  = (x-one)/(x+one); 
	    }
	} else {
	    if (ix < 0x40038000) {	/* |x| < 2.4375 */
		id = 2; x  = (x-1.5)/(one+1.5*x);
	    } else {			/* 2.4375 <= |x| < 2^66 */
		id = 3; x  = -1.0/x;
	    }
	}}
    /* end of argument reduction */
	z = x*x;
	w = z*z;
    /* break sum from i=0 to 10 aT[i]z**(i+1) into odd and even poly */
	s1 = z*(aT[0]+w*(aT[2]+w*(aT[4]+w*(aT[6]+w*(aT[8]+w*aT[10])))));
	s2 = w*(aT[1]+w*(aT[3]+w*(aT[5]+w*(aT[7]+w*aT[9]))));
	if (id<0) return x - x*(s1+s2);
	else {
	    z = atanhi[id] - ((x*(s1+s2) - atanlo[id]) - x);
	    return (hx<0)? -z:z;
	}
}


BOTH double my_acos(double x)
{
	double z,p,q,r,w,s,c,df;
	int hx,ix;
	hx = __HI(x);
	ix = hx&0x7fffffff;
	if(ix>=0x3ff00000) {	/* |x| >= 1 */
	    if(((ix-0x3ff00000)|__LO(x))==0) {	/* |x|==1 */
		if(hx>0) return 0.0;		/* acos(1) = 0  */
		else return pi+2.0*pio2_lo;	/* acos(-1)= pi */
	    }
	    return (x-x)/(x-x);		/* acos(|x|>1) is NaN */
	}
	if(ix<0x3fe00000) {	/* |x| < 0.5 */
	    if(ix<=0x3c600000) return pio2_hi+pio2_lo;/*if|x|<2**-57*/
	    z = x*x;
	    p = z*(pS0+z*(pS1+z*(pS2+z*(pS3+z*(pS4+z*pS5)))));
	    q = one+z*(qS1+z*(qS2+z*(qS3+z*qS4)));
	    r = p/q;
	    return pio2_hi - (x - (pio2_lo-x*r));
	} else  if (hx<0) {		/* x < -0.5 */
	    z = (one+x)*0.5;
	    p = z*(pS0+z*(pS1+z*(pS2+z*(pS3+z*(pS4+z*pS5)))));
	    q = one+z*(qS1+z*(qS2+z*(qS3+z*qS4)));
	    s = sqrt(z);
	    r = p/q;
	    w = r*s-pio2_lo;
	    return pi - 2.0*(s+w);
	} else {			/* x > 0.5 */
	    z = (one-x)*0.5;
	    s = sqrt(z);
	    df = s;
	    __LO(df) = 0;
	    c  = (z-df*df)/(s+df);
	    p = z*(pS0+z*(pS1+z*(pS2+z*(pS3+z*(pS4+z*pS5)))));
	    q = one+z*(qS1+z*(qS2+z*(qS3+z*qS4)));
	    r = p/q;
	    w = r*s+c;
	    return 2.0*(df+w);
	}
}


BOTH double my_atan2(double y, double x)
{  
	double z;
	int k,m,hx,hy,ix,iy;
	unsigned lx,ly;

	hx = __HI(x); ix = hx&0x7fffffff;
	lx = __LO(x);
	hy = __HI(y); iy = hy&0x7fffffff;
	ly = __LO(y);
	if(((ix|((lx|-lx)>>31))>0x7ff00000)||
	   ((iy|((ly|-ly)>>31))>0x7ff00000))	/* x or y is NaN */
	   return x+y;
	if((hx-0x3ff00000|lx)==0) return my_atan(y);   /* x=1.0 */
	m = ((hy>>31)&1)|((hx>>30)&2);	/* 2*sign(x)+sign(y) */

    /* when y = 0 */
	if((iy|ly)==0) {
	    switch(m) {
		case 0: 
		case 1: return y; 	/* atan(+-0,+anything)=+-0 */
		case 2: return  pi+tiny;/* atan(+0,-anything) = pi */
		case 3: return -pi-tiny;/* atan(-0,-anything) =-pi */
	    }
	}
    /* when x = 0 */
	if((ix|lx)==0) return (hy<0)?  -pi_o_2-tiny: pi_o_2+tiny;
	    
    /* when x is INF */
	if(ix==0x7ff00000) {
	    if(iy==0x7ff00000) {
		switch(m) {
		    case 0: return  pi_o_4+tiny;/* atan(+INF,+INF) */
		    case 1: return -pi_o_4-tiny;/* atan(-INF,+INF) */
		    case 2: return  3.0*pi_o_4+tiny;/*atan(+INF,-INF)*/
		    case 3: return -3.0*pi_o_4-tiny;/*atan(-INF,-INF)*/
		}
	    } else {
		switch(m) {
		    case 0: return  zero  ;	/* atan(+...,+INF) */
		    case 1: return -zero  ;	/* atan(-...,+INF) */
		    case 2: return  pi+tiny  ;	/* atan(+...,-INF) */
		    case 3: return -pi-tiny  ;	/* atan(-...,-INF) */
		}
	    }
	}
    /* when y is INF */
	if(iy==0x7ff00000) return (hy<0)? -pi_o_2-tiny: pi_o_2+tiny;

    /* compute y/x */
	k = (iy-ix)>>20;
	if(k > 60) z=pi_o_2+0.5*pi_lo; 	/* |y/x| >  2**60 */
	else if(hx<0&&k<-60) z=0.0; 	/* |y|/x < -2**60 */
	else z=my_atan(fabs(y/x));		/* safe to do y/x */
	switch (m) {
	    case 0: return       z  ;	/* atan(+,+) */
	    case 1: __HI(z) ^= 0x80000000;
		    return       z  ;	/* atan(-,+) */
	    case 2: return  pi-(z-pi_lo);/* atan(+,-) */
	    default: /* case 3 */
	    	    return  (z-pi_lo)-pi;/* atan(-,-) */
	}
}