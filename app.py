# 問診AI

import streamlit as st
import openai
import json
import numpy as np
import pandas as pd
import time
import requests

# API keys
openai.api_key = "openai"
DEEPSEEK_API_KEY = "deepseek"  # Replace with your DeepSeek API key

###医学知識###
# 症状リスト
columns_dictionary_1 = ['胸痛', '呼吸困難', '腹痛', '発熱', 'めまい', '頭痛', '意識障害', '動悸', 'けいれん', '吐血', '下血', '血尿', '腰痛', '背部痛', '浮腫', '発疹', '関節痛', '四肢のしびれ', '四肢の麻痺', '外傷', '不眠', '鼻汁', '咽頭痛', '咳嗽', '倦怠感']
# 追加質問リスト
next_question_map = {
    "腹痛": {
        1: [
            "お腹のどの部分が痛みますか？（例: 右上腹部、左下腹部、全体的になど）",
            "痛みの感じ方について教えてください。鋭い痛み（キリキリ）、鈍い痛み（ズーン）、刺すような痛み（チクチク）など、どのような痛みでしょうか？",
            "この痛みと一緒に下痢や嘔吐の症状はありますか？もしある場合、頻度や回数についても教えてください。",
            "これまでの人生で経験した最も強い痛みを10点とすると、今の痛みは何点くらいでしょうか？",
            "痛みはずっと続いていますか？それとも波があり、強くなったり弱くなったりしますか？",
            "いつからこの痛みを感じていますか？（例: 今日の朝から、2日前の夜から、1週間前から など）",
            "最近、生ものや鍋物を食べましたか？特に傷んだ可能性のあるものを食べていませんか？",
            "最後に便が出たのはいつですか？排便の状態（硬さや色、血が混じっているかなど）についても教えてください。",
            "最後に食事を取ったのはいつですか？その際に食べたものも教えてください。",
            "ご家族や友人などで周囲に同様の症状の方はいらっしゃいませんか？",
            "これまでにお腹の手術を受けたことがありますか？もしある場合、どんな手術を受けたか教えてください。",
            "水分は摂取できますか？しんどくて、水を飲むことも困難ではないですか？",
            "歩いた時に、お腹に響く感じはしますか？"
        ],
        0: []
    },
    "胸痛": {
        1: [
            "痛みは急激に始まりましたか？",
            "胸のどのあたりが痛みますか？（みぞおち、真ん中、左、右など）",
            "どのような痛みでしょうか？（締め付けられるような痛み、刺すような痛み、焼けるような痛みなど）",
            "いつからこのような痛みを自覚していますか(10分前ですか？3日前ですか？)",
            "人生最大の痛みを10点とすると、今の痛みは何点くらいですか？",
            "痛いところを手で押すと痛みは強くなりますか？それともほとんど変わらないですか？",
            "痛みの範囲は10円玉程度ですか、それとも手のひら大程度ですか？",
            "痛みは階段を登る時と、座っているときとでどのタイミングで強くなりますか？",
            "痛みは肩や腕、背中、あごなどに広がりますか？",
            "他に冷や汗はありますか？",
            "息苦しい感じ、吐き気などの症状はありますか？"
        ],
        0: []
    },
    "呼吸困難": {
        1: [
            "息苦しさはいつから感じていますか？（急に始まった、徐々に悪化しているなど）",
            "安静にしていても息苦しいですか？運動時や横になるときなど、どの場面で悪化しますか？",
            "夜眠れますか？横になるとと息が苦しくなって体を起こすと息苦しさが改善したりしませんか？",
            # "息苦しさの程度を10段階で表すとどのくらいですか？",
            "他に咳や発熱、胸痛、アレルギーの有無など、思い当たる症状やきっかけはありますか？",
            "肺や心臓の持病や喘息はありますか？",
            "思い当たる原因はありますか？"
        ],
        0: []
    },
    "発熱": {
        1: [
            "いつから熱がありますか？（今日の朝から、1ヶ月前からなど）",
            "最高で何度くらいの熱が出ていますか？",
            "発熱以外の症状はありますか？（咳、のどの痛み、鼻水、腹痛、下痢、発疹、腰痛など）",
            "思い当たるきっかけ（人混みや海外渡航、周囲の感染状況など）はありますか？",
            "解熱剤を使用した場合、効果はありますか？"
        ],
        0: []
    },
    "めまい": {
        1: [
            "どのようなめまいですか？（ぐるぐる、ふわふわ感、立ちくらみなど）",
            "手足が動かしくくないですか？",
            "頭痛はありませんか？",
            "めまいはいつからですか？きっかけはありましたか？",
            "めまいの際に耳鳴りや難聴、吐き気、嘔吐はありますか？",
            "立ち上がったときや頭を動かしたときなど、姿勢で症状は変化しますか？",
            "過去にも同じようなめまいの経験はありますか？",
        ],
        0: []
    },
    "頭痛": {
        1: [
            "痛みの程度は人生最大の痛みではないですか？最大を10点とした時に何点くらいですか？",
            # "頭のどの部分が痛みますか？（前頭部、後頭部、側頭部、全体など）",
            # "痛みの性質はどうですか？（ズキズキ、締め付けられるような、重い感じ、刺すような痛みなど）",
            "いつからですかか？（10分前, 1時間前, 1週間前など）",
            "3分程度で突然痛みが強くなりましたか?1時間程度以上かけて徐々に痛くなりましたか?",
            "手足が動かしくい、話しにくいなどありませんか?",
            # "頭痛と一緒に吐き気や嘔吐、めまい、光や音に敏感になる症状はありますか？",
            # "頭痛薬は飲んでいますか？効果はどうですか？"
        ],
        0: []
    },
    "意識障害": {
        1: [
            "いつから意識がもうろうとしたり、失神したりすることがありましたか？",
            "意識障害の直前に何かきっかけ（強い痛み、暑さ、息苦しさなど）はありましたか？",
            "意識を失った際、痙攣や失禁はありましたか？",
            "意識障害から回復したあと、どのような状態でしたか？（すぐ普通に戻った、しばらくぼーっとしていたなど）",
            "過去にも同様のエピソードはありましたか？"
        ],
        0: []
    },
    "動悸": {
        1: [
            "動悸はどんなときに起こりますか？（安静時、運動時、ストレス時など）",
            "動悸はどのくらい続きますか？（数秒、数分、数十分など）",
            "動悸と同時に胸が痛くなったりや息苦しさ、めまいなどはありますか？",
            "過去に心臓病や不整脈を指摘されたことはありますか？",
            "カフェインの摂取や喫煙習慣はありますか？",
            "失神(一時的に意識を失うこと)の経験はありますか？"
        ],
        0: []
    },
    "けいれん": {
        1: [
            "全身がけいれんしましたか？それとも手足など一部だけですか？",
            "どのくらいの時間けいれんが続きましたか？（数秒、数分など）",
            "けいれん中、意識はありましたか？意識が飛んでいたなどはありますか？",
            "けいれんの原因として思い当たることはありますか？（疲労、発熱、持病など）",
            "過去に同じようなけいれんを起こしたことはありますか？（診断名など）"
        ],
        0: []
    },
    "吐血": {
        1: [
            "吐血に気づいたのはいつですか？（突然、徐々になど）",
            "血はどのくらいの量でしたか？（コップ何杯分、少量など）",
            "血の色や状態はどうでしたか？（真っ赤、黒っぽい、コーヒー残渣様など）",
            "吐血の前に胃痛や胸焼け、吐き気などはありましたか？",
            "過去に胃潰瘍や肝硬変、食道静脈瘤などを指摘されたことはありますか？"
        ],
        0: []
    },
    "下血": {
        1: [
            "いつから下血に気づきましたか？",
            "血の色や便の状態はどうですか？（真っ赤、黒っぽいタール状、混ざっているなど）",
            "腹痛や下痢はありますか？",
            "過去に痔や潰瘍性大腸炎、大腸ポリープなどと診断されたことはありますか？",
            "同じような症状を経験したことはありますか？"
        ],
        0: []
    },
    "血尿": {
        1: [
            "いつ血尿に気づきましたか？（突然、検査で分かったなど）",
            "尿の色はどのようでしたか？（ピンク、赤色、茶色など）",
            "排尿時に痛みや違和感はありますか？（焼けるような痛み、残尿感など）",
            "他に発熱や腰の痛み、むくみなどの症状はありますか？",
            "過去に腎臓や尿路系の病気を指摘されたことはありますか？"
        ],
        0: []
    },
    "腰痛": {
        1: [
            "いつから腰痛がありますか？（急に始まった、慢性的など）",
            "痛みの原因として思い当たることはありますか？（重い物を持った、長時間同じ姿勢など）",
            "どのような痛みですか？（鋭い痛み、鈍い痛み、筋肉痛のような痛みなど）",
            "動作によって痛みは変わりますか？（曲げる、ひねる、座る、立ち上がるなど）",
            "腰痛以外に、しびれや発熱、下肢の痛みなどはありますか？",
            "大便や尿を漏らしたりしていませんか？"
        ],
        0: []
    },
    "背部痛": {
        1: [
            "背中のどの部分が痛みますか？（上部、中部、下部など）",
            "いつから痛み始めましたか？きっかけ（転倒、運動、長時間のデスクワークなど）はありますか？",
            "どのような痛みですか？（鈍い痛み、刺すような痛み、焼けるような痛みなど）",
            "日常生活で不便を感じることはありますか？（立ち上がりにくい、寝返りがつらいなど）",
            "他にしびれや発熱、胸痛などの症状はありますか？"
        ],
        0: []
    },
    "浮腫": {
        1: [
            "むくみは体のどこに出ていますか？（足、顔、手など）",
            "いつからむくみが気になるようになりましたか？（朝起きたとき、夕方など）",
            "むくみを押すと跡が残りますか？",
            "普段の水分や塩分の摂取量は多いですか？",
            "心臓や腎臓、肝臓などの持病、あるいは服用薬はありますか？"
        ],
        0: []
    },
    "発疹": {
        1: [
            "発疹はどの部分に出ていますか？（顔、腕、胴体など）",
            "いつから出始めましたか？（突然、徐々に）",
            "発疹の形状や特徴はどうですか？（赤い斑点、水ぶくれ、かさぶたなど）",
            "かゆみや痛み、熱感はありますか？",
            "似たような発疹が過去にもありましたか？アレルギー歴はありますか？",
            "最近虫刺されするような環境、例えば山の中に行ったりしましたか？"
        ],
        0: []
    },
    "関節痛": {
        1: [
            "どの関節が痛みますか？（膝、手首、指、肩など）",
            "いつから痛み始めましたか？（急性か慢性か）",
            "痛みの性質や特徴は？（ズキズキ、腫れ、熱感、こわばりなど）",
            "どのタイミングで痛みが強くなりますか？（朝、動作開始時など）",
            "過去に関節のケガやリウマチなどを指摘されたことはありますか？"
        ],
        0: []
    },
    "四肢のしびれ": {
        1: [
            "どの部分にしびれを感じますか？（手先、足先、片側だけなど）",
            "しびれはいつからですか？きっかけはありましたか？",
            "しびれ以外に痛みや筋力低下、感覚麻痺などはありますか？",
            "しびれは持続的ですか？それとも断続的ですか？",
            "過去に神経や血管の病気、ヘルニアなどを指摘されたことはありますか？"
        ],
        0: []
    },
    "四肢の麻痺": {
        1: [
            "どの部位に麻痺を感じますか？（右手、左足など）",
            "いつから麻痺を感じるようになりましたか？（急に、徐々に）",
            "麻痺とともに痛みやしびれはありますか？",
            "麻痺は進行していますか、それとも回復傾向がありますか？",
            "過去に脳や神経の病気（脳卒中など）を指摘されたことはありますか？"
        ],
        0: []
    },
    "外傷": {
        1: [
            "いつ、どのようにケガをしましたか？（転倒、交通事故、スポーツなど）",
            "どの部位をケガしましたか？（頭部、腕、脚、背中など）",
            "出血や痛み、腫れ、変形などの症状はありますか？",
            "受傷後、すぐに病院を受診しましたか？応急処置はしましたか？",
            "同じ部位を以前にもケガしたことはありますか？"
        ],
        0: []
    },
    "不眠": {
        1: [
            "いつから眠れませんか？",
            "寝付きが悪いですか？それとも寝ている間に目が覚めますか？",
            "睡眠不足の原因として思い当たることはありますか？（ストレス、不安、カフェインの摂取など）",
            "横になると、息が苦しくなって体を起こすと息苦しさが改善したりしませんか？",
            "夜に何度もトイレに起きていませんか？その場合はトイレに行く回数も教えてください",
            "死にたいと思うようなことはありますか？",
            "日中の活動に支障が出ていますか？"
        ],
        0: []
    },
    "鼻汁": {
        1: [
            "いつから鼻水が出ていますか？（急に始まった、徐々になど）",
            "鼻水の色や粘度はどうですか？（透明、黄色っぽい、粘り気がある、血が混じるなど）",
            "くしゃみや鼻づまり、のどの痛みなど他の症状はありますか？",
            "花粉症やアレルギー性鼻炎などの持病はありますか？",
            "周囲に同じような症状の人がいますか？"
        ],
        0: []
    },
    "咽頭痛": {
        1: [
            "いつから喉が痛いですか？（急性か徐々にか）",
            "痛みはどんなときに強く感じますか？（飲み込むとき、話すときなど）",
            "のどの腫れや赤み、発熱、咳、鼻水など他の症状はありますか？",
            "これまでに同様の症状を繰り返したことはありますか？（扁桃炎など）",
            "唾液を飲み込めないほどの喉の痛みではないですか？",
            "喉が痛くて水がのめなくなっていませんか？"
        ],
        0: []
    },
    "咳嗽": {
        1: [
            "いつから咳が出ていますか？（突然、徐々になど）",
            "咳の性質はどうですか？（乾いた咳、痰が絡む咳、夜間に強くなるなど）",
            "痰がある場合、その色や粘度はどうですか？（透明、黄色や緑っぽいなど）",
            "発熱や息苦しさ、胸痛などを伴いますか？",
            "喫煙歴やアレルギー、喘息などの持病はありますか？"
        ],
        0: []
    },
    "倦怠感": {
        1: [
            "倦怠感はいつから感じますか？（急に始まった、徐々に強くなったなど）",
            "倦怠感の程度はどのくらいですか？（日常生活に支障が出るほどなど）",
            "他に発熱、食欲不振、体重減少などの症状はありますか？",
            "睡眠時間は十分ですか？生活リズムは安定していますか？",
            "ストレスや精神的な負担（仕事、人間関係など）は強く感じていますか？"
        ],
        0: []
    },
    "嘔吐": {
        1: [
            "いつから嘔吐がはじまりましたか？（急に始まったか、徐々に増えたかなど）",
            "胸が痛かったり、頭が痛かったりしませんか？",
            "嘔吐の頻度はどのくらいですか？（1日に何回、1時間おきなど）",
            "嘔吐する前に吐き気や腹痛、胸やけなどの症状はありましたか？",
            "吐いたものの色や形状はどうでしたか？（透明、白色、黄色、緑色、茶色、血が混じっているなど）",
            "吐いた後に楽になりますか？それとも気分の悪さが続きますか？",
            "他に下痢や発熱、めまいなどの症状はありますか？",
            "脱水症状（口の渇き、尿量の減少など）はありませんか？",
            "食事は取れていますか？水分補給はできていますか？",
            "過去に同じような嘔吐の症状がありましたか？そのときの原因は何でしたか？"
        ],
        0: []
    },
    "咳嗽": {
        1: [
            "いつから咳が出ていますか？（突然、徐々になど）",
            "咳の性質はどうですか？（乾いた咳、痰が絡む咳、夜間に強くなるなど）",
            "痰がある場合、その色や粘度はどうですか？（透明、黄色や緑っぽいなど）",
            "発熱や息苦しさ、胸痛などを伴いますか？",
            "喫煙歴やアレルギー、喘息などの持病はありますか？"
        ],
        0: []
    },
}

# レッドフラッグサイン
red_flag_sign_map = {
    "胸痛": [
        "冷や汗",
        "急激な発症",
        "ペインスケール>7",
        "呼吸困難を伴う",
        "胸を締め付けられるような痛み",
        # "ショックバイタル（血圧低下・頻脈）",
        # "意識レベル低下"
    ],
    "呼吸困難": [
        "SpO2<90%",
        "チアノーゼ",
        "呼吸回数の著明な増加や努力呼吸",
        "意識レベルの低下",
        "血圧低下やショック症状"
    ],
    "腹痛": [
        "突然の激痛",
        "バイタル異常（血圧低下・頻脈）",
        "板状硬腹（強い筋性防御）",
        "血便や吐血を伴う",
        "発熱や悪寒"
    ],
    "発熱": [
        # "39℃以上の高熱",
        "意識レベルの低下",
        # "皮下出血班・皮膚粘膜症状（重症感染症など）",
        # "呼吸・循環動態の著明な異常",
        # "強い倦怠感や脱力"
    ],
    "めまい": [
        "突然の意識消失や失神",
        "強い頭痛や嘔吐を伴う",
        "言語障害や視野異常",
        "歩行困難や片麻痺",
        "頸部痛（脳血管障害の可能性）"
    ],
    "頭痛": [
        "突然の激しい頭痛（サンダークラップヘッドエイク）",
        "意識障害を伴う",
        "けいれんを伴う",
        "発熱や項部硬直（髄膜炎の可能性）",
        "神経脱落症状（片麻痺・感覚障害など）"
    ],
    "意識障害": [
        "呼びかけや痛みに反応しない",
        "バイタル異常（特に呼吸回数や脈拍）",
        "けいれんの既往または観察",
        "頭部外傷の既往や外傷痕",
        "薬物・アルコール摂取歴による中毒の疑い"
    ],
    "動悸": [
        "胸痛を伴う",
        "意識消失や失神",
        "脈の乱れ（不整脈の疑い）",
        "呼吸困難の併発",
        "ショックバイタル（血圧低下・頻脈）"
    ],
    "けいれん": [
        "意識レベルの著明な低下",
        "連続または頻回にけいれん（てんかん重積状態）",
        "頭部外傷の痕跡",
        "発熱や項部硬直（髄膜炎・脳炎など）",
        "電解質異常の疑い（既往や検査所見）"
    ],
    "吐血": [
        "大量の吐血によるショックバイタル",
        "黒色便を伴う（上部消化管出血の可能性）",
        "急激な貧血症状（めまい・ふらつき）",
        "肝硬変や胃潰瘍などの重篤既往",
        "意識レベルの低下"
    ],
    "下血": [
        "大量の下血",
        # "ショックバイタル（血圧低下・頻脈）",
        # "黒色便（上部消化管出血の疑い）",
        # "強い腹痛や肛門痛",
        "著明な貧血症状（めまい・ふらつき）"
    ],
    "血尿": {
        "鮮紅色尿やワインカラー尿など明らかな血尿",
        "腰痛や下腹部痛を伴う",
        "排尿困難や頻尿",
        "悪寒を伴う発熱（腎盂腎炎など）",
        "ショック症状（大量出血時）"
    },
    "腰痛": {
        "急性の激痛（ぎっくり腰以外の深刻な可能性）",
        "下肢の感覚異常や運動麻痺",
        "膀胱直腸障害（排尿・排便障害）",
        "発熱や体重減少（感染・悪性腫瘍の疑い）",
        "バイタル異常を伴う"
    },
    "背部痛": {
        "突然の激しい痛み（大動脈解離の疑い）",
        "血圧の左右差やバイタル異常",
        "胸痛・腹痛への放散痛",
        "ショック症状",
        "大動脈瘤などの既往歴"
    },
    "浮腫": {
        "急激な体重増加",
        "呼吸困難（心不全の疑い）",
        "高度のむくみ（全身性）",
        "尿量減少（腎不全の疑い）",
        "明らかな心不全症状（起坐呼吸など）"
    },
    "発疹": {
        "広範囲かつ急速に増加",
        "水疱形成や粘膜病変（SJSなど重症皮膚障害）",
        "強い疼痛やかゆみ",
        "高熱や全身倦怠感",
        "ショック症状（アナフィラキシーの可能性）"
    },
    "関節痛": {
        "急性の変形や強い腫脹",
        "可動域の著明な制限",
        "激しい発赤や発熱を伴う関節",
        "明らかな外傷既往",
        "全身症状（倦怠感、体重減少など）"
    },
    "四肢のしびれ": {
        "急性発症",
        "進行性の麻痺",
        "排尿・排便障害（脊髄病変の可能性）",
        "高度の痛みや感覚障害",
        "椎間板ヘルニアや脳卒中の既往"
    },
    "四肢の麻痺": {
        "急激な発症（脳卒中など）",
        "意識障害や失語など中枢神経症状を伴う",
        "感覚障害の併発",
        "強い頭痛やめまいを伴う",
        "不整脈や心房細動の既往"
    },
    "外傷": {
        "頭部外傷での意識障害",
        "大量出血や開放骨折",
        "呼吸・循環動態の不安定",
        "頸椎損傷の疑い（首の痛みや四肢麻痺）",
        "複数部位の重傷"
    },
    "不眠": {
        # "長期化（数週間以上持続）",
        "精神症状（妄想・幻覚・抑うつなど）を伴う",
        "重度の倦怠感や自殺念慮",
        "睡眠時無呼吸症候群を疑う所見（著明ないびき・呼吸停止）",
        "昼間の過度の眠気で生活に支障"
    },
    "鼻汁": {
        1: [
            "血性鼻汁",
            # "大量または悪臭のある膿性鼻汁",
            # "顔面痛や発熱（重症副鼻腔炎の可能性）",
            # "長期化による嗅覚障害",
            # "外傷後の脳脊髄液漏れの疑い"
        ],
        0: []
    },
    "咽頭痛": {
        1: [
            "嚥下困難",
            "呼吸困難",
            "39℃以上の高熱",
            "唾液の飲み込みができず涎が多量（咽頭蓋炎の疑い）",
            "顎下や頸部リンパ節の強い腫脹"
        ],
        0: []
    },
    "咳嗽": {
        1: [
            "呼吸困難を伴う",
            "大量の血痰",
            "3週間以上持続（慢性咳嗽）",
            "胸痛やバイタル異常の併発",
            "高熱や体重減少（肺炎・結核の可能性）"
        ],
        0: []
    },
    "倦怠感": {
        1: [
            "急激な悪化",
            "起き上がれないほどの重症度",
            "高熱や呼吸苦の併発",
            "意識障害や重度のめまいを伴う",
            "著明な体重減少（悪性疾患や重症感染症の疑い）"
        ],
        0: []
    }
}
# 診療科のリスト
depertment_list = ['内科', '整形外科', '外科', '皮膚科', '眼科', '耳鼻咽喉科', '小児科', '産婦人科', '泌尿器科', '神経内科', '精神科', '心療内科', '救急科',  '歯科', '口腔外科', '呼吸器内科', '循環器内科', '消化器内科', '内分泌代謝内科', '腎臓内科', '血液内科', 'リウマチ科']

###関数定義###
# 文字を1文字ずつ表示
def typewrite(text: str, speed=0.05):
    typed_text = ""
    message_placeholder = st.empty()
    for char in text:
        typed_text += char
        message_placeholder.markdown(typed_text)
        time.sleep(speed)

# using chat GPT 4o
def chat_to_gpt_4o(prompt):
    MODEL = "gpt-4o-2024-08-06"
    completion = openai.ChatCompletion.create(
        model=MODEL,
        temperature=0,
        top_p=0.5,
        messages=[
            {"role": "system", "content": "You are a great assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def chat_to_gpt_4o_temperature_0(prompt):
    MODEL = "gpt-4o-2024-08-06"
    completion = openai.ChatCompletion.create(
        model=MODEL,
        temperature=0,
        top_p=0.5,
        messages=[
            {"role": "system", "content": "You are a great assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def chat_to_deepseek(prompt):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.session_state.get('deepseek_api_key', DEEPSEEK_API_KEY)}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a great assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "top_p": 0.5
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 401:
            st.error("DeepSeek APIキーが無効です。サイドバーから正しいAPIキーを入力してください。")
            return None
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        st.error(f"DeepSeek APIへの接続に失敗しました: {str(e)}\nAPIキーを確認してください。")
        return None

def chat_to_deepseek_temperature_0(prompt):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.session_state.get('deepseek_api_key', DEEPSEEK_API_KEY)}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a great assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "top_p": 0.5
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 401:
            st.error("DeepSeek APIキーが無効です。サイドバーから正しいAPIキーを入力してください。")
            return None
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.Timeout:
        st.error("DeepSeek APIの応答がタイムアウトしました（10秒以内に返答がありませんでした）。時間をおいて再度お試しください。")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"DeepSeek APIへの接続に失敗しました: {str(e)}\nAPIキーを確認してください。")
        return None

# Modify the existing functions to use either GPT-4 or DeepSeek based on a parameter
def chat_with_model(prompt, model="gpt4", temperature=0):
    try:
        if model == "gpt4":
            if temperature == 0:
                return chat_to_gpt_4o_temperature_0(prompt)
            return chat_to_gpt_4o(prompt)
        elif model == "deepseek":
            if temperature == 0:
                return chat_to_deepseek_temperature_0(prompt)
            return chat_to_deepseek(prompt)
        else:
            raise ValueError(f"Unsupported model: {model}")
    except Exception as e:
        st.error(f"APIエラーが発生しました: {str(e)}\nAPIキーが正しく設定されているか確認してください。")
        return None

# extract symptom from patient's comment
def out_put_dictionary(patients_comment, columns_dictionary=columns_dictionary_1):
    prompt = f"""
    あなたは優秀な医師です。下記は患者の発言です。
    患者の発言から症状リストにある症状があるかどうかを確認して、
    それぞれの症状に対してdictionary (JSON) の形式で 0/1 で返してください。
    Constrains:
    - どの症状も該当しなさそうであれば、症状をよく見て最も近いものにしてください。例えば、胃もたれや生理痛は腹痛としてください。
    - 出力は必ずJSON形式の辞書で返してください。
    - 必ず1つ以上の症状を1にしてください。全て0にはしないでください。
    - 出力例: {{ \"腹痛\": 1, \"発熱\": 0 }}
    - JSON以外の文章や説明、コードブロック、改行は一切含めないでください。
    - もし症状リストに該当する内容がなければ、最も近い症状を1にしてください。
    # 患者の発言
    {patients_comment}
    # 症状リスト
    {columns_dictionary}
    """

    str_output = chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)
    if str_output is None:
        st.error("症状の分析に失敗しました。APIキーを確認してください。")
        st.stop()
    # コードブロックや余計な文字列を除去
    str_output = str_output.strip()
    if str_output.startswith('```json'):
        str_output = str_output.replace('```json', '').replace('```', '').strip()
    elif str_output.startswith('```'):
        str_output = str_output.replace('```', '').strip()
    try:
        dict_output = json.loads(str_output)
        return dict_output
    except json.JSONDecodeError:
        st.error("APIからの応答を解析できませんでした。\nAIからの返答: " + str(str_output))
        st.stop()

def extract_additional_symptom(patients_comment, columns_dictionary=columns_dictionary_1):
    prompt = f"""
    あなたは優秀な医師です。下記は患者の問診票で、質問とその回答が書かれています。
    患者の発言から症状リストにある症状があるかどうかを確認して、
    それぞれの症状に対してdictionary (JSON) の形式で 0/1 で返してください。

    # 患者の問診票
    {patients_comment}

    # 症状リスト
    {columns_dictionary}

    # 制約
    - 必ず dictionaryのみを出力してください。
    - 例: {{ "腹痛": 0, "咽頭痛": 1 }}
    - 上記以外の解説や文章は出力しないでください。
    - 改行もしないでください。
    """

    str_output = chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)
    dict_output = json.loads(str_output)

    return dict_output

# extract next question from additional question list
def get_additional_question(symptom_dict):
    next_question=[]

    for symptom, question_dict in next_question_map.items():
        if symptom in symptom_dict:
            value = symptom_dict[symptom]
            if value in question_dict:
                next_question.extend(question_dict[value])
    return next_question

# get next question from patient's comment
def get_next_question(patients_comment):
    symptom_dict = out_put_dictionary(patients_comment)
    next_question = get_additional_question(symptom_dict)
    return next_question

# すでに発言内に書いてあれば、追加質問をなくする
def create_case_dict(patients_comment, next_question):
    case_dict = {}
    for i in range(len(next_question)):
        prompt = f"""
        患者の発言の中に、質問への回答が含まれていればその回答を抜き出してdictionary形式にしてください。
        もし患者の発言に質問に対する回答が全く含まれないか、正確に記載されれいなければ0にしてください。
        患者の発言: {patients_comment},
        質問: {next_question[i]},
        制約:
        回答になっていいるかどうか悩ましい場合は0としてください。
        0もしくは、文字列で返してください
        - 例: 0,
        - 例: カフェインのとりすぎです。
        - 上記以外の解説や文章は出力しないでください。
        - 改行もしないでください。
        """
        str_response = chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)
        if str_response is None:
            st.error("質問への回答の分析に失敗しました。APIキーを確認してください。")
            st.stop()
        
        case_dict[next_question[i]] = str_response
    return case_dict

def make_question_and_dictionary(patients_comment, columns_dictionary=columns_dictionary_1):
    # まずは抽出
    symptom_dictionary = out_put_dictionary(patients_comment, columns_dictionary)
    if symptom_dictionary is None:
        st.error("問診データの取得に失敗しました。最初からやり直してください。")
        st.stop()
    
    # 次の質問リストを作成
    next_question_list = get_additional_question(symptom_dictionary)
    if not next_question_list:
        return {}, symptom_dictionary
    
    # 患者の発言内に既に答えがあるかどうかをチェックした辞書を作る
    case_dict = create_case_dict(patients_comment=patients_comment, next_question=next_question_list)
    if case_dict is None:
        st.error("問診データの取得に失敗しました。最初からやり直してください。")
        st.stop()
        
    return case_dict, symptom_dictionary


# サマリ作成と確認
def make_summary(query_anwer_dictionary):
    prompt = f'''患者に患者が記載した問診票の内容に誤りがないかどうか確認したい。
    以下の問診表の内容を一続きの自然な文章に要約して患者に内容の間違えがないかどうかを確認し、間違えや気になる点があれば教えてもらってください。
    問診票: {query_anwer_dictionary}
  　制約: 患者が読む文章なので「あなたの症状をまとめましたので確認してください」から開始してください。'''
    return chat_with_model(prompt, model=st.session_state["selected_model"])

# 最初のサマリと患者の追加の発言をサマライズ
def make_final_summary(summary, patients_additional_comment):
    prompt = f'''以下の患者の訴えのまとめと、患者の補足から最終的なサマリーを作成してください。
    サマリー: {summary}
    患者の補足:{patients_additional_comment}
    制約: 箇条書きを使用しないで記載してください。
    情報の順番は自然な形にしてください。
    情報量を減らしたり追加をしないでください。
    患者に見せるものなので、丁寧かつ中学生でもわかるように記載してください。'''
    return chat_with_model(prompt, model=st.session_state["selected_model"])

# レッドフラッグサインを抽出
def extract_red_flag_signs(structured_symptom):
    symptom_list = [k for k, v in structured_symptom.items() if v == 1]
    red_flag_sign_list = []
    for symptom in symptom_list:
        red_flag_sign_list.append(red_flag_sign_map[symptom])
    return red_flag_sign_list

# レッドフラッグサインの有無をもとに、緊急性の有無を判断させる。
def evaluate_urgency(summary, red_flag_sign_list):
    prompt = f'''以下の患者サマリと、救急車を呼ぶべき危険な兆候です。
    緊急性が高い指標の有無を確認して救急車を呼んだ方がよいかどうかを判断してください。
    危険な兆候があれば緊急性ありと判断してください。
    患者サマリ:{summary}
    緊急性が高い指標: {red_flag_sign_list}
    緊急性が低い場合の出力例: あなたは胸が痛い症状でお困りですね。胸が痛い場合に危険性が高い症状は冷や汗や今でも胸が痛いこと、、、です。
    これらの症状は認めないので今すぐに救急車を呼ぶほどではありませんが、症状の変化があった場合はその限りではないので、また教えてください。
    緊急性が高い場合の出力例: あなたは胸が痛い症状でお困りですね。胸が痛い場合に高い症状は冷や汗や今でも胸が痛いこと、、、です。
    あなたはこれらのうち、冷や汗を認めており、心筋梗塞なども疑われるため緊急性が高い可能性があります。今すぐに救急車を呼ぶことをおすすめします'''
    return chat_with_model(prompt, model=st.session_state["selected_model"])

# 受診すべき診療科を考える
def make_decision(summary_ver2):
    prompt = f'''以下の問診表の内容から症状の根源的な原因を考えて想定される疾患を2-3個考えてください。
    また、その疾患から受診するべき診療科目とその次に受診するべき科目を診療科リストから選んで教えてください。
    問診票: {summary_ver2}
    診療科リスト: {depertment_list}
    制約: 100文字以内で箇条書きを使用しないで記載してください。
    情報の順番は自然な形にしてください。
    情報量を減らしたり追加をしないでください。
    患者に見せるものなので、丁寧かつ中学生でもわかるように記載してください。
    受診推奨科は3つまでにしてください。
    '''
    return chat_with_model(prompt, model=st.session_state["selected_model"])

def hospital_iwami_decision(summary, depertment_assessement):
    prompt=f'''あなたは医師であり、現在夜間救急当直をしており、救急隊から患者受入可能かどうかを判断しています。
    以下の患者サマリと、推奨診療科に関するアセスメントを参照し、受け入れ可能基準と照らし合わせて、患者受け入れ可能かどうかを判断してください。
    患者サマリ: {summary}
    受診推奨科に関するアセスメント: {depertment_assessement}
    受け入れ可能基準: 内科系疾患の場合は受け入れ困難。
    回答例: 岩見病院ですが、本日整形外科疾患しか受け入れをしていないです。
'''
    return chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)

def hospital_watanabe_decision(summary, depertment_assessement):
    prompt=f'''あなたは医師であり、現在夜間救急当直をしており、救急隊から患者受入可能かどうかを判断しています。
    以下の患者サマリと、推奨診療科に関するアセスメントを参照し、受け入れ可能基準と照らし合わせて、患者受け入れ可能かどうかを判断してください。
    患者サマリ: {summary}
    受診推奨科に関するアセスメント: {depertment_assessement}
    受け入れ可能基準: 肺炎など内科系疾患の場合は受け入れ困難。ただし、血圧低下やショック、手術が必要になる可能性がある腹痛などの患者は受け入れが困難。
    回答例: 渡辺病院ですが、本日内科一般、外科治療を要さない心疾患のみ受け入れ可能です。そのため腹部手術が必要な患者さんの受け入れは難しいです。
'''
    return chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)

def hospital_kikuoka_decision(summary, depertment_assessement):
    prompt=f'''あなたは医師であり、現在夜間救急当直をしており、救急隊から患者受入可能かどうかを判断しています。
    以下の患者サマリと、推奨診療科に関するアセスメントを参照し、受け入れ可能基準と照らし合わせて、患者受け入れ可能かどうかを判断してください。
    患者サマリ: {summary}
    受診推奨科に関するアセスメント: {depertment_assessement}
    受け入れ可能基準: 救急全般受け入れ可能。肺炎や血圧低下のない発熱などは可能であれば他の病院で受け入れたい。
    回答例: 菊岡病院ですが、3次医療施設なのでどのような患者でも受け入れが可能です。ただ、医療リソースの最適化もあるので軽症な患者は他の病院で受け入れていただけると助かります。
'''
    return chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)

def hospital_kato_decision(summary, depertment_assessement):
    prompt=f'''あなたは医師であり、現在夜間救急当直をしており、救急隊から患者受入可能かどうかを判断しています。
    以下の患者サマリと、推奨診療科に関するアセスメントを参照し、受け入れ可能基準と照らし合わせて、患者受け入れ可能かどうかを判断してください。
    患者サマリ: {summary}
    受診推奨科に関するアセスメント: {depertment_assessement}
    受け入れ可能基準: 内科患者、外科患者の受け入れができず、精神疾患のみが疑われる場合のみ受け入れ可能
    回答例: 加藤病院ですが、本日精神疾患患者しか受け入れをしていないです。
'''
    return chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)

def hospital_saku_decision(summary, depertment_assessement):
    prompt=f'''あなたは医師であり、現在夜間救急当直をしており、救急隊から患者受入可能かどうかを判断しています。
    以下の患者サマリと、推奨診療科に関するアセスメントを参照し、受け入れ可能基準と照らし合わせて、患者受け入れ可能かどうかを判断してください。
    患者サマリ: {summary}
    受診推奨科に関するアセスメント: {depertment_assessement}
    受け入れ可能基準: 内科一般、外科治療を要さない心疾患のみ受け入れ可能、腹部手術が必要な患者の受け入れは難しい。
    回答例: こちら朔病院ですが、本日内科一般、外科治療を要さない心疾患のみ受け入れ可能です。そのため腹部手術が必要な患者さんの受け入れは難しいです。
'''
    return chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)

def analyze_stomach_pain(patient_input):
    """
    Analyzes stomach pain characteristics and location from patient input
    Returns a dictionary with pain details
    """
    prompt = f"""
    You are a medical professional analyzing a patient's description of stomach pain.
    Please analyze the following patient input and return a JSON dictionary with the following information:
    - location: Where exactly is the pain located? (e.g., upper right, lower left, whole abdomen)
    - type: What type of pain is it? (e.g., sharp, dull, cramping)
    - severity: On a scale of 1-10, how severe is the pain?
    - duration: How long has the pain been present?
    - associated_symptoms: Any associated symptoms like nausea, vomiting, diarrhea?
    
    Patient input: {patient_input}
    
    Return only a JSON dictionary, no additional text.
    Example format:
    {{
        "location": "upper right abdomen",
        "type": "sharp",
        "severity": 7,
        "duration": "2 hours",
        "associated_symptoms": ["nausea", "vomiting"]
    }}
    """
    
    try:
        response = chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)
        if response:
            return json.loads(response)
    except Exception as e:
        st.error(f"Error analyzing stomach pain: {str(e)}")
    return None

def analyze_body_pain(patient_input):
    """
    Analyzes any type of body pain from patient input with an empathetic approach
    Returns a dictionary with pain details including confidence levels and recovery tips
    """
    prompt = f"""
    You are a caring and empathetic medical professional analyzing a patient's description of pain.
    Please analyze the following patient input and return a JSON dictionary with the following information:
    - pain_type: What type of pain is it? (e.g., sharp, dull, throbbing, burning)
    - location: Where exactly is the pain located? (e.g., head, back, chest, abdomen, limbs)
    - severity: On a scale of 1-10, how severe is the pain?
    - duration: How long has the pain been present?
    - possible_causes: List of possible causes with confidence levels (0-100%)
    - associated_symptoms: Any associated symptoms
    - confidence_level: Overall confidence in the analysis (0-100%)
    - empathetic_note: A brief, caring note about the patient's situation
    - recovery_tips: 2-3 short, practical tips for immediate relief or recovery
    - follow_up_questions: 2-3 relevant questions to better understand the situation
    
    Patient input: {patient_input}
    
    Return only a JSON dictionary, no additional text.
    Example format:
    {{
        "pain_type": "sharp",
        "location": "lower back",
        "severity": 7,
        "duration": "3 days",
        "possible_causes": [
            {{"cause": "muscle strain", "confidence": 80}},
            {{"cause": "herniated disc", "confidence": 40}},
            {{"cause": "kidney stone", "confidence": 20}}
        ],
        "associated_symptoms": ["stiffness", "limited mobility"],
        "confidence_level": 85,
        "empathetic_note": "I understand this pain must be quite uncomfortable for you. Let's work together to find the best way to help you feel better.",
        "recovery_tips": [
            "軽いストレッチを試してみてください。痛みを悪化させない程度にゆっくりと体を動かしましょう。",
            "患部を温めると痛みが和らぐ可能性があります。温かいタオルや湯たんぽを使用してみてください。",
            "無理な姿勢を避け、適度な休息を取るように心がけてください。"
        ],
        "follow_up_questions": [
            "痛みは特定の動作や姿勢で強くなりますか？",
            "これまでに同じような痛みを経験したことはありますか？",
            "痛み以外に気になる症状はありますか？"
        ]
    }}
    """
    
    try:
        response = chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)
        if response:
            return json.loads(response)
    except Exception as e:
        st.error(f"申し訳ありませんが、痛みの分析中に問題が発生しました。もう一度お試しください。")
    return None

###メイン処理###

import streamlit as st
from deepseek_api import DeepSeekAPI

# ページ設定
st.set_page_config(
    page_title="問診AI",
    page_icon="🤖",
    layout="wide"
)

# APIキーの設定
API_KEY = "sk-fb5ed929cd134354b20d4557c194e651"

# DeepSeek APIクライアントの初期化
client = DeepSeekAPI(API_KEY)

# タイトル
st.title("問診AI")
st.text("正確な問診をするAIです。")

# セッションで管理するステート
if "step" not in st.session_state:
    st.session_state["step"] = 0  # 0から開始するように変更
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = None
if "api_keys" not in st.session_state:
    st.session_state["api_keys"] = {
        "openai": "",
        "deepseek": "sk-fb5ed929cd134354b20d4557c194e651"
    }
if "stomach_pain_analysis" not in st.session_state:
    st.session_state["stomach_pain_analysis"] = None
if "body_pain_analysis" not in st.session_state:
    st.session_state["body_pain_analysis"] = None
if "patients_summary_ver1" not in st.session_state:
    st.session_state["patients_summary_ver1"] = ""

# サイドバーにモデル設定を配置
with st.sidebar:
    st.markdown("### フィードバック")
    st.markdown("ご意見・ご要望があれば以下までお願いいたします。")
    st.markdown("[アンケート](https://forms.gle/MuRWMHM23wPwPAQH8)")
    st.markdown("[GitHub Issues](https://github.com/yusukewatanabe1208/test/issues)")
    st.markdown("---")  # Add a separator line
    
    st.subheader("AIモデルの設定")
    
    # モデル選択（ドロップダウンリスト）
    model_choice = st.selectbox(
        "利用するAIモデルを選択してください",
        ["GPT-4", "DeepSeek"],
        index=1
    )
    
    # 選択されたモデルに応じたAPIキー入力
    if model_choice == "GPT-4":
        api_key = st.text_input(
            "OpenAI APIキー",
            type="password"
        )
    else:
        api_key = st.text_input(
            "DeepSeek APIキー",
            type="password",
            value="sk-fb5ed929cd134354b20d4557c194e651"
        )
    
    if st.button("設定を保存して開始"):
        if api_key:
            # モデルとAPIキーを保存
            st.session_state["selected_model"] = "gpt4" if model_choice == "GPT-4" else "deepseek"
            
            # APIキーを保存
            if model_choice == "GPT-4":
                st.session_state["api_keys"]["openai"] = api_key
                openai.api_key = api_key
            else:
                st.session_state["api_keys"]["deepseek"] = api_key
                DEEPSEEK_API_KEY = api_key
                # セッションステートにも保存
                st.session_state["deepseek_api_key"] = api_key
            
            st.session_state.step = 1
            st.rerun()
        else:
            st.error("APIキーを入力してください。")

    # サイドバーの一番下に終了ボタンを追加
    st.markdown("---")  # Add a separator line
    if st.button("終了", type="primary"):
        st.session_state.step = 0
        st.session_state["messages"] = []
        st.session_state["selected_model"] = None
        st.session_state["api_keys"] = {
            "openai": "",
            "deepseek": ""
        }
        st.rerun()

# メインコンテンツ
if st.session_state.step == 0:
    st.info("左側のサイドバーからAIモデルを選択し、APIキーを入力してください。")
    for idx, msg in enumerate(st.session_state.get("messages", [])):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    st.chat_input("メッセージを入力してください")

elif st.session_state.step == 1:
    # チャット履歴の表示
    for idx, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # チャット入力ボックス
    prompt = st.chat_input("メッセージを入力してください")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.session_state["patients_first_comment"] = prompt
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    # step2でもチャット履歴と入力ボックスを必ず表示
    for idx, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    prompt = st.chat_input("メッセージを入力してください")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        # --- 直前の症状リストを保存・比較 ---
        prev_symptoms = st.session_state.get("current_symptoms", set())
        # 新しい症状抽出
        case_dict, symptom_dictionary = make_question_and_dictionary(prompt)
        # 新たに1になった症状だけ抽出
        new_symptoms = set([k for k, v in symptom_dictionary.items() if v == 1]) - prev_symptoms
        st.session_state["current_symptoms"] = prev_symptoms | new_symptoms
        # 新規症状があれば、その症状ごとに1問だけ質問
        if new_symptoms:
            # 新規症状が複数でも、最初の1つだけに質問（初回安定化のため）
            symptom = list(new_symptoms)[0]
            followup_prompt = f"ユーザーは「{symptom}」と述べました。その症状について日本語で1つだけ追加質問してください。質問文のみを出力してください。解説や説明は不要です。"
            followup_question = chat_with_model(followup_prompt, model=st.session_state["selected_model"], temperature=0)
            if not followup_question or followup_question.strip() == "":
                followup_question = f"{symptom}について、もう少し詳しく教えてください。例えば、いつから続いているか、どのくらいの強さかなどを教えてください。"
            with st.chat_message("assistant"):
                st.write(followup_question)
                st.session_state["messages"].append({"role": "assistant", "content": followup_question})
            st.rerun()  # 追加: 直後に画面を再描画して必ず応答を表示
            # pending_questionsに残りの症状があれば追加
            st.session_state["pending_questions"] = list(new_symptoms - {symptom}) + list(case_dict.keys())
        else:
            # 既存症状の追加発言や新規症状がなければ通常フロー
            if "pending_questions" not in st.session_state or not st.session_state["pending_questions"]:
                st.session_state["pending_questions"] = list(case_dict.keys())
            if st.session_state["pending_questions"]:
                next_q = st.session_state["pending_questions"].pop(0)
                with st.chat_message("assistant"):
                    st.write(next_q)
                    st.session_state["messages"].append({"role": "assistant", "content": next_q})
            else:
                summary = make_summary(symptom_dictionary)
                with st.chat_message("assistant"):
                    st.write("追加の質問はありません。症状の説明を続けてください。\n\n【まとめ】\n" + str(summary))
                    st.session_state["messages"].append({"role": "assistant", "content": "追加の質問はありません。症状の説明を続けてください。\n\n【まとめ】\n" + str(summary)})