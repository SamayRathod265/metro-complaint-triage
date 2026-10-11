"""Generate training_data.json: synthetic multilingual metro complaints for the Naive Bayes layer.

Each complaint = a language-specific *frame* (greeting, station mention, request to fix…) wrapped around a
department-specific *problem* phrase. Problems use the everyday words real passengers use (lift, toilet,
ticket, guard…). None of these sentences appear in eval_complaints.json or sample_complaints.json — the test
for that lives in test_app.py.

Run:  python tools/generate_training_data.py   (deterministic, seed below)
"""
import json
import os
import random

SEED = 7
PER_PROBLEM = 3  # frames sampled per problem phrase
STATIONS = ["Thaltej", "Vastral Gam", "Kalupur", "Motera Stadium", "Old High Court", "Gandhinagar Sector 1",
            "Apparel Park", "Shahpur", "Ghee Kanta", "AEC", "Gyaspur Depot", "Sabarmati"]

FRAMES = {
    "English": ["{p}", "{p}.", "{p} at {s}.", "At {s} station, {p}.", "{p}. Please fix this.", "Hello, {p}.",
                "{p} since this morning.", "{p}, very bad experience.", "Sir, {p}. Kindly look into it.",
                "{p}!! Nobody is responding.", "Complaint: {p}", "{p} near gate 2."],
    "Hindi": ["{p}", "{p}।", "{s} स्टेशन पर {p}।", "{p}, कृपया ध्यान दें।", "{p}, जल्दी ठीक करें।",
              "नमस्ते, {p}।", "{p}, कोई सुनने वाला नहीं है।", "शिकायत: {p}", "{p}, बहुत परेशानी हो रही है।"],
    "Gujarati": ["{p}", "{p}.", "{s} સ્ટેશન પર {p}.", "{p}, મહેરબાની કરીને ધ્યાન આપો.", "{p}, જલ્દી ઉકેલ લાવો.",
                 "નમસ્તે, {p}.", "{p}, કોઈ જવાબ નથી આપતું.", "ફરિયાદ: {p}", "{p}, બહુ તકલીફ પડે છે."],
    "Gujlish": ["{p}", "{p}.", "{s} station par {p}.", "{p}, please dhyan aapo.", "{p}, jaldi kaik karo.",
                "Kem cho, {p}.", "{p}, koi jawab nathi aapatu.", "{p}, bahu taklif pade che.", "{s} ma {p}."],
    "Hinglish": ["{p}", "{p}.", "{s} station pe {p}.", "{p}, please dhyan do.", "{p}, jaldi theek karo.",
                 "Namaste, {p}.", "{p}, koi sun nahi raha.", "{p}, bahut pareshani ho rahi hai.", "{s} mein {p}."],
}

# department -> language -> [(problem phrase, urgency)]
PROBLEMS = {
    "safety": {
        "English": [("smoke is coming out of the coach", 5), ("a passenger collapsed on the platform", 5),
                    ("someone is harassing women in the ladies coach", 5), ("a child slipped into the gap near the train", 5),
                    ("live wire hanging near the entry stairs", 5), ("a man with a knife was threatening people", 5),
                    ("train moved with the doors open", 5), ("there was a fight and someone is bleeding", 5),
                    ("abandoned bag kept near the security check for an hour", 5), ("sparks from the electric panel", 5),
                    ("people pushing dangerously at the crowded gate", 4), ("no guard at the station exit late at night", 3),
                    ("cctv camera not working and phones getting stolen", 3), ("floor near platform edge is wet and slippery", 4)],
        "Hindi": [("कोच में धुआं निकल रहा है", 5), ("एक यात्री बेहोश हो गया है", 5), ("महिला कोच में लड़कियों को परेशान किया जा रहा है", 5),
                  ("बिजली का तार खुला लटक रहा है", 5), ("एक आदमी चाकू दिखाकर डरा रहा था", 5), ("दरवाज़े खुले रहते हुए ट्रेन चल पड़ी", 5),
                  ("लावारिस बैग पड़ा हुआ है", 5), ("गेट पर बहुत धक्का-मुक्की हो रही है", 4), ("रात में सुरक्षा गार्ड नहीं होता", 3),
                  ("सीसीटीवी बंद है और मोबाइल चोरी हो रहे हैं", 3)],
        "Gujarati": [("ડબ્બામાંથી ધુમાડો નીકળે છે", 5), ("એક મુસાફર બેભાન થઈને પડ્યો છે", 5), ("મહિલા ડબ્બામાં છોકરીઓને હેરાન કરે છે", 5),
                     ("વીજળીનો વાયર ખુલ્લો લટકે છે", 5), ("એક માણસ છરી બતાવીને ધમકાવતો હતો", 5), ("દરવાજા ખુલ્લા હતા અને ટ્રેન ચાલી", 5),
                     ("બિનવારસી બેગ પડી છે", 5), ("ગેટ પર ભારે ધક્કામુક્કી થાય છે", 4), ("રાત્રે કોઈ સુરક્ષા ગાર્ડ નથી હોતો", 3),
                     ("સીસીટીવી બંધ છે અને મોબાઈલ ચોરાય છે", 3)],
        "Gujlish": [("coach mathi dhumado nikle che", 5), ("ek musafar behosh thai gayo che", 5),
                    ("ladies coach ma chhokrio ne heran kare che", 5), ("current no wire khullo latke che", 5),
                    ("ek manas chhari batavi ne dhamkavto hato", 5), ("darvaja khulla hata ne train chali gai", 5),
                    ("binvarsi bag padi che", 5), ("gate par bahu dhakka-mukki thay che", 4),
                    ("raat na koi security guard nathi hota", 3), ("cctv bandh che ane mobile chorai jay che", 3)],
        "Hinglish": [("coach se dhuan nikal raha hai", 5), ("ek passenger behosh ho gaya hai", 5),
                     ("ladies coach mein ladkiyon ko pareshan kar rahe hain", 5), ("bijli ka wire khula latak raha hai", 5),
                     ("ek aadmi chaaku dikha ke dara raha tha", 5), ("darwaze khule the aur train chal padi", 5),
                     ("lawaris bag pada hua hai", 5), ("gate pe bahut dhakka-mukki ho rahi hai", 4),
                     ("raat ko koi security guard nahi hota", 3), ("cctv band hai aur mobile chori ho rahe hain", 3)],
    },
    "lifts": {
        "English": [("the lift is not working", 4), ("escalator stopped working", 3), ("elevator is out of service", 4),
                    ("lift door gets stuck halfway", 3), ("escalator is shaking and making noise", 4),
                    ("no lift for wheelchair users", 4), ("lift buttons are not responding", 3),
                    ("escalator has been switched off for a week", 3), ("lift takes forever to arrive", 2),
                    ("lift fan and light not working", 2), ("down escalator is closed for repairs again", 3),
                    ("the elevator display shows wrong floor", 2)],
        "Hindi": [("लिफ्ट काम नहीं कर रही", 4), ("एस्केलेटर बंद पड़ा है", 3), ("लिफ्ट का दरवाज़ा अटक जाता है", 3),
                  ("एस्केलेटर से अजीब आवाज़ आती है", 4), ("व्हीलचेयर के लिए लिफ्ट नहीं चल रही", 4), ("लिफ्ट के बटन नहीं दबते", 3),
                  ("एक हफ्ते से एस्केलेटर बंद है", 3), ("लिफ्ट बहुत देर से आती है", 2), ("लिफ्ट में पंखा और लाइट बंद है", 2)],
        "Gujarati": [("લિફ્ટ ચાલતી નથી", 4), ("એસ્કેલેટર બંધ પડ્યું છે", 3), ("લિફ્ટનો દરવાજો અટકી જાય છે", 3),
                     ("એસ્કેલેટરમાંથી અવાજ આવે છે", 4), ("વ્હીલચેર માટે લિફ્ટ ચાલુ નથી", 4), ("લિફ્ટના બટન કામ નથી કરતા", 3),
                     ("એક અઠવાડિયાથી એસ્કેલેટર બંધ છે", 3), ("લિફ્ટ આવવામાં બહુ વાર લાગે છે", 2), ("લિફ્ટમાં પંખો અને લાઈટ બંધ છે", 2),
                     ("એલિવેટર બગડેલું છે", 3)],
        "Gujlish": [("lift chalti nathi", 4), ("escalator bandh padyu che", 3), ("lift no darvajo atki jay che", 3),
                    ("escalator mathi avaj aave che", 4), ("wheelchair mate lift chalu nathi", 4), ("lift na button kaam nathi karta", 3),
                    ("ek athvadiya thi escalator bandh che", 3), ("lift aavva ma bahu var lage che", 2),
                    ("lift ma pankho ane light bandh che", 2), ("elevator bagdelu che", 3)],
        "Hinglish": [("lift kaam nahi kar rahi", 4), ("escalator band pada hai", 3), ("lift ka darwaza atak jata hai", 3),
                     ("escalator se ajeeb awaaz aa rahi hai", 4), ("wheelchair ke liye lift nahi chal rahi", 4),
                     ("lift ke button kaam nahi karte", 3), ("ek hafte se escalator band hai", 3),
                     ("lift aane mein bahut time lagta hai", 2), ("lift mein pankha aur light band hai", 2), ("elevator kharab hai", 3)],
    },
    "cleanliness": {
        "English": [("the toilet is very dirty", 2), ("washroom smells terrible", 2), ("garbage lying on the platform", 2),
                    ("dustbins are overflowing", 2), ("spit and paan marks on the walls", 2), ("coach floor is full of litter", 2),
                    ("no water in the restroom", 2), ("cockroaches and rats in the coach", 3), ("someone vomited in the coach", 3),
                    ("water cooler is filthy", 3), ("no soap or tissue in the washroom", 2), ("stairs are dusty and nobody cleans them", 2)],
        "Hindi": [("शौचालय बहुत गंदा है", 2), ("बाथरूम से बदबू आ रही है", 2), ("प्लेटफॉर्म पर कचरा पड़ा है", 2),
                  ("कूड़ेदान भर गए हैं", 2), ("दीवारों पर पान और थूक के दाग हैं", 2), ("कोच में गंदगी फैली है", 2),
                  ("टॉयलेट में पानी नहीं है", 2), ("कोच में चूहे और कॉकरोच हैं", 3), ("सफाई कोई नहीं करता", 2)],
        "Gujarati": [("શૌચાલય ખૂબ ગંદુ છે", 2), ("બાથરૂમમાંથી દુર્ગંધ આવે છે", 2), ("પ્લેટફોર્મ પર કચરો પડ્યો છે", 2),
                     ("કચરાપેટીઓ ઉભરાઈ ગઈ છે", 2), ("દીવાલો પર પાન અને થૂંકના ડાઘ છે", 2), ("ડબ્બામાં ગંદકી ફેલાયેલી છે", 2),
                     ("ટોયલેટમાં પાણી નથી", 2), ("ડબ્બામાં ઉંદર અને વંદા છે", 3), ("કોઈ સફાઈ નથી કરતું", 2)],
        "Gujlish": [("toilet bahu gandu che", 2), ("bathroom mathi gandh aave che", 2), ("platform par kachro padyo che", 2),
                    ("dustbin bharai gaya che", 2), ("diwal par paan ni pichkari che", 2), ("coach ma gandagi che", 2),
                    ("toilet ma paani nathi", 2), ("coach ma undar ane vanda che", 3), ("koi safai nathi kartu", 2)],
        "Hinglish": [("toilet bahut ganda hai", 2), ("bathroom se badbu aa rahi hai", 2), ("platform pe kachra pada hai", 2),
                     ("dustbin bhar gaye hain", 2), ("deewaron pe paan ke daag hain", 2), ("coach mein gandagi hai", 2),
                     ("toilet mein paani nahi hai", 2), ("coach mein chuhe aur cockroach hain", 3), ("koi safai nahi karta", 2)],
    },
    "fare": {
        "English": [("ticket machine ate my money", 3), ("money deducted twice from my card", 3), ("QR code ticket not accepted at gate", 3),
                    ("charged extra fare for a short trip", 3), ("card recharge not reflecting", 3), ("token not dispensed after payment", 3),
                    ("refund still not received", 2), ("UPI payment failed but money debited", 3),
                    ("student concession not applied", 2), ("counter did not return my change", 2),
                    ("app shows wrong fare", 2), ("gate did not open after scanning ticket", 3)],
        "Hindi": [("टिकट मशीन ने पैसे ले लिए", 3), ("कार्ड से दो बार पैसे कटे", 3), ("क्यूआर टिकट गेट पर नहीं चला", 3),
                  ("कम दूरी का ज़्यादा किराया लिया", 3), ("रिचार्ज कार्ड में नहीं दिखा", 3), ("पैसे देने के बाद टोकन नहीं मिला", 3),
                  ("रिफंड अभी तक नहीं मिला", 2), ("यूपीआई से पैसे कटे पर टिकट नहीं बना", 3), ("छात्र छूट नहीं मिली", 2)],
        "Gujarati": [("ટિકિટ મશીને પૈસા લઈ લીધા", 3), ("કાર્ડમાંથી બે વખત પૈસા કપાયા", 3), ("ક્યુઆર ટિકિટ ગેટ પર ચાલતી નથી", 3),
                     ("ટૂંકી મુસાફરીનું વધારે ભાડું લીધું", 3), ("રિચાર્જ કાર્ડમાં દેખાતું નથી", 3), ("પૈસા આપ્યા પછી ટોકન ન મળ્યું", 3),
                     ("રિફંડ હજી મળ્યું નથી", 2), ("યુપીઆઈથી પૈસા કપાયા પણ ટિકિટ ન બની", 3), ("વિદ્યાર્થી છૂટ ન મળી", 2)],
        "Gujlish": [("ticket machine e paisa lai lidha", 3), ("card mathi be var paisa kapaya", 3), ("QR ticket gate par chalti nathi", 3),
                    ("tunki musafari nu vadhare bhadu lidhu", 3), ("recharge card ma dekhatu nathi", 3), ("paisa aapya pachi token na malyo", 3),
                    ("refund haju malyu nathi", 2), ("UPI thi paisa kapaya pan ticket na bani", 3), ("student discount na malyu", 2)],
        "Hinglish": [("ticket machine ne paise kha liye", 3), ("card se do baar paise kate", 3), ("QR ticket gate pe nahi chala", 3),
                     ("chhoti journey ka zyada kiraya liya", 3), ("recharge card mein nahi dikha", 3), ("paise dene ke baad token nahi mila", 3),
                     ("refund abhi tak nahi aaya", 2), ("UPI se paise kate par ticket nahi bana", 3), ("student discount nahi mila", 2)],
    },
    "staff_behaviour": {
        "English": [("the security guard was rude to me", 3), ("staff shouted at an old lady", 3), ("counter clerk ignored everyone", 2),
                    ("staff refused to help a disabled passenger", 4), ("guard pushed me without reason", 4),
                    ("staff used abusive language", 3), ("ticket counter lady was busy on her phone", 2),
                    ("employee misbehaved with my family", 3), ("staff refused to register my complaint", 2),
                    ("helpdesk person gave a very arrogant reply", 2)],
        "Hindi": [("सुरक्षा गार्ड ने बदतमीज़ी की", 3), ("स्टाफ ने बुजुर्ग महिला पर चिल्लाया", 3), ("काउंटर वाला सबको अनदेखा करता है", 2),
                  ("कर्मचारी ने विकलांग यात्री की मदद नहीं की", 4), ("गार्ड ने बिना वजह धक्का दिया", 4), ("स्टाफ ने गाली दी", 3),
                  ("काउंटर वाली मैडम फोन पर व्यस्त थी", 2), ("कर्मचारी का व्यवहार बहुत खराब था", 3)],
        "Gujarati": [("સિક્યોરિટી ગાર્ડે ખરાબ વર્તન કર્યું", 3), ("સ્ટાફે વૃદ્ધ મહિલા પર બૂમ પાડી", 3), ("કાઉન્ટરવાળો બધાને અવગણે છે", 2),
                     ("કર્મચારીએ દિવ્યાંગ મુસાફરને મદદ ન કરી", 4), ("ગાર્ડે વગર કારણે ધક્કો માર્યો", 4), ("સ્ટાફે ગાળો આપી", 3),
                     ("કાઉન્ટર પરની બહેન ફોનમાં વ્યસ્ત હતી", 2), ("કર્મચારીનું વર્તન ઉદ્ધત હતું", 3)],
        "Gujlish": [("security guard e kharab vartan karyu", 3), ("staff e vrudhh ben par bum padi", 3), ("counter valo badha ne avgane che", 2),
                    ("karmachari e divyang musafar ne madad na kari", 4), ("guard e vagar karane dhakko maryo", 4), ("staff e gaalo aapi", 3),
                    ("counter par ni ben phone ma vyast hati", 2), ("staff nu vartan uddhat hatu", 3)],
        "Hinglish": [("security guard ne badtameezi ki", 3), ("staff ne buzurg aunty pe chillaya", 3), ("counter wala sabko ignore karta hai", 2),
                     ("staff ne disabled passenger ki madad nahi ki", 4), ("guard ne bina wajah dhakka mara", 4), ("staff ne gaali di", 3),
                     ("counter wali madam phone pe busy thi", 2), ("staff ka behaviour bahut kharab tha", 3)],
    },
    "other": {
        "English": [("please run more trains in the evening", 1), ("station wifi is not working", 1), ("need more parking space", 1),
                    ("add Gujarati announcements", 1), ("train was late with no information", 2), ("please start service earlier in the morning", 1),
                    ("need a feeder bus from the station", 1), ("train timings in the app are wrong", 1), ("signage is confusing", 1),
                    ("lost my bag, where is lost and found", 2), ("please add mobile charging points", 1), ("no drinking water facility", 2)],
        "Hindi": [("शाम को ज़्यादा ट्रेनें चलाइए", 1), ("स्टेशन का वाईफाई नहीं चलता", 1), ("पार्किंग की जगह कम है", 1),
                  ("गुजराती में घोषणा कीजिए", 1), ("ट्रेन बिना सूचना के लेट थी", 2), ("सुबह जल्दी सेवा शुरू कीजिए", 1),
                  ("स्टेशन से बस की सुविधा चाहिए", 1), ("ऐप में ट्रेन का समय गलत है", 1), ("पीने के पानी की सुविधा नहीं है", 2)],
        "Gujarati": [("સાંજે વધારે ટ્રેન ચલાવો", 1), ("સ્ટેશનનું વાઈફાઈ ચાલતું નથી", 1), ("પાર્કિંગની જગ્યા ઓછી છે", 1),
                     ("ગુજરાતીમાં જાહેરાત કરો", 1), ("ટ્રેન કોઈ સૂચના વગર મોડી હતી", 2), ("સવારે વહેલી સેવા શરૂ કરો", 1),
                     ("સ્ટેશનથી બસની સુવિધા જોઈએ", 1), ("એપમાં ટ્રેનનો સમય ખોટો છે", 1), ("પીવાના પાણીની સુવિધા નથી", 2)],
        "Gujlish": [("saanje vadhare train chalavo", 1), ("station nu wifi chaltu nathi", 1), ("parking ni jagya ochhi che", 1),
                    ("Gujarati ma announcement karo", 1), ("train koi suchna vagar modi hati", 2), ("savare vaheli seva sharu karo", 1),
                    ("station thi bus ni suvidha joiye", 1), ("app ma train no time khoto che", 1), ("pivana paani ni suvidha nathi", 2)],
        "Hinglish": [("shaam ko zyada trains chalao", 1), ("station ka wifi nahi chalta", 1), ("parking ki jagah kam hai", 1),
                     ("Gujarati mein announcement karo", 1), ("train bina information ke late thi", 2), ("subah jaldi service shuru karo", 1),
                     ("station se bus ki facility chahiye", 1), ("app mein train ka time galat hai", 1), ("peene ke paani ki suvidha nahi hai", 2)],
    },
}


def generate(seed=SEED):
    rng = random.Random(seed)
    rows = []
    for dept, by_lang in PROBLEMS.items():
        for lang, problems in by_lang.items():
            for phrase, urgency in problems:
                for frame in rng.sample(FRAMES[lang], PER_PROBLEM):
                    p = phrase[0].upper() + phrase[1:] if frame.startswith("{p}") and lang == "English" else phrase
                    rows.append({"station": rng.choice(STATIONS), "language": lang,
                                 "text": frame.format(p=p, s=rng.choice(STATIONS)),
                                 "gold_department": dept, "gold_urgency": urgency})
    rng.shuffle(rows)
    seen, out = set(), []
    for r in rows:  # drop accidental duplicates
        if r["text"] not in seen:
            seen.add(r["text"])
            out.append({"id": 1000 + len(out), **r})
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = generate()
    json.dump(data, open(os.path.join(here, "training_data.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {len(data)} rows to training_data.json")
