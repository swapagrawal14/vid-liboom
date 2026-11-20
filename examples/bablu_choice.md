# Bablu ka Personal Mausam: Tum Chuno!

## Scene: Park mein Dome | id=dome | bg=sky | mood=day
BABLU [proud]: Dekho Didi, mera naya personal weather dome! Jo mausam chahiye, wahi milega.
DIDI [sideeye]: Bablu, pichhli baar tumhare gadget ne fridge ko cloud bana diya tha.
UNCLE [read]: Beta, mausam se panga mat lo. Mausam kisi ka nahi hota.
BABLU [smug]: Uncle, ab hota hai. Sirf mera.
(Bablu ke sir ke upar ek chamakta hua kaanch ka dome)

## Scene: Dome ke andar Baarish | id=rain | bg=sky | mood=rain
BABLU [flail]: Arre! Dome ke andar baarish kyun ho rahi hai?
DIDI [giggle]: Bahar dhoop, andar baarish. Wah, personal mausam!
UNCLE [shrug]: Maine kaha tha na, mausam ki apni marzi hoti hai.
BABLU [worried]: Remote pe teen button hain, aur sab gile ho gaye!
? Bablu kya kare?
-> Remote ka button dabao | goto=button
-> Didi se madad maango | goto=didi

## Scene: Remote ka Button | id=button | bg=sky | mood=sunset
BABLU [grope]: Ek button pe suraj bana hai, ek pe bijli. Kaunsa dabaun?
DIDI [startled]: Bijli wala mat dabana! Mat dabana!
UNCLE [lookup]: Aasmaan bhi saans rok ke dekh raha hai, beta.
? Kaunsa button?
-> Suraj wala button | goto=sunny
-> Bijli wala button | goto=storm

## Scene: Didi ka Jugaad | id=didi | bg=sky | mood=day | end
DIDI [idle]: Laao, remote mujhe do. Isme ek setting hai: sabke liye mausam.
BABLU [slump]: Matlab mera personal mausam ab personal nahi rahega?
UNCLE [zen]: Jo mausam baant liya, wahi sabse achha mausam hai.
DIDI [photo]: Pure park mein halki thandi hawa. Ek selfie banti hai!
BABLU [wave]: Theek hai, theek hai. Sab ka mausam, sab ki khushi!

## Scene: Suraj wala Button | id=sunny | bg=sky | mood=sunset | end
BABLU [proud]: Dekho! Dome mein garma garam dhoop, aur meri chhoti si sunset!
DIDI [giggle]: Bablu, tumhara chehra tamatar jaisa laal ho gaya hai.
BABLU [flinch]: Aah! Bahut zyada dhoop! Sunscreen kahan hai?
UNCLE [hands]: Beta, har cheez balance mein achhi lagti hai. Dhoop bhi.
BABLU [smug]: Kal se setting ho jayegi: aadhi dhoop, aadhi chhaanv.

## Scene: Bijli wala Button | id=storm | bg=sky | mood=night | end
BABLU [flail]: Baap re! Dome mein toofan aa gaya!
UNCLE [armsup]: Bijli kadki, aur mera akhbaar ud gaya!
DIDI [startled]: Bablu, dome se bahar niklo, jaldi!
BABLU [slump]: Mera dome ab ek chhota sa badal ban ke ud raha hai.
UNCLE [zen]: Koi baat nahi beta. Aaj raat taare dekhte hain, bina kisi dome ke.
