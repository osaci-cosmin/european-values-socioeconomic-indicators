
# V1 Survey Weighting Policy
# Politica de ponderare V1

---

## English

### Purpose

The ESS respondent-level data are based on complex probability samples.
Therefore, unweighted sample statistics are not treated as final population
estimates in this project.

The project preserves the original ESS weighting and sample-design variables:

- `dweight`
- `pspwght`
- `pweight`
- `anweight`
- `psu`
- `stratum`

No ESS weight is manually altered.

---

### Weight validation

For the V1 ESS Round 11 analytical sample:

- 49,210 respondents are included.
- 29 countries are included.
- All four ESS weight variables are present for every respondent.
- No weight has a zero or negative value.
- The ESS analysis-weight identity was empirically confirmed:

`anweight = pspwght × pweight`

with a maximum observed numerical difference of:

`0.000000000000`

The population weight (`pweight`) is constant within each country.

Therefore, within a single country, weighted point estimates calculated with
`pspwght` and `anweight` are numerically equivalent.

This equivalence was verified directly for the main analytical variables.

---

### Primary V1 weight

The primary weight for descriptive point estimates is:

`anweight`

This provides one consistent weighting rule across the project.

---

### Country-level estimates

Country-level means and proportions use:

`anweight`

Examples include:

- mean Left-Right self-placement;
- Left / Centre / Right shares;
- mean Social Trust;
- mean centered Human Values scores.

Because `pweight` is constant within each country, these point estimates are
equivalent to estimates calculated using `pspwght`.

---

### Regional and pooled estimates

V1 does not report regional or cross-country pooled respondent-level estimates.

For future analyses combining respondents from multiple countries, `anweight`
is a candidate weight when the target estimand requires countries to contribute
in proportion to their population sizes. The final weighting choice must be
justified in relation to the target population and research question.

Unlike purely within-country estimates, the population-size component matters
when the aim is to estimate characteristics of a pooled population.

Using the appropriate population component avoids treating countries with
similar ESS sample sizes but very different populations as if they contributed
equally to a population-level estimate.

---

### Confidence intervals and standard errors

Point-estimate weights alone are not sufficient for survey-design-based
uncertainty estimation.

In V1, standard errors and confidence intervals were computed for selected
country-level ESS survey estimates.

The uncertainty-estimation workflow incorporates relevant ESS sampling-design
information, including:

- `psu`
- `stratum`
- `anweight`

These survey-level uncertainty estimates are distinct from the uncertainty
of country-level Pearson and Spearman correlation coefficients.

Sampling uncertainty from ESS country estimates is not fully propagated
into the 36 cross-country correlation coefficients.

Consequently, the reported country-level correlations should not be
interpreted as accounting for all uncertainty associated with the
underlying ESS survey estimates.

---

### Country-level analytical dataset

After respondent-level weighting, the project creates a country-level dataset
with one observation per country.

Once the unit of analysis becomes the country, respondent weights are not
automatically reused in country-level models.

For example:

`GDP PPS ↔ Mean Social Trust`

is a relationship between countries.

In the 36 V1 country-level Pearson and Spearman correlations, each country
with available data for a given variable pair contributes equally. ESS survey
weights are used to estimate country-level aggregates, but are not reapplied
to the correlations between countries.

The weighting strategy for other country-level statistical models must therefore
be chosen according to the country-level estimand rather than inherited
mechanically from the respondent-level survey.

---

### Multilevel modelling

Survey weights will not be inserted mechanically into multilevel models.

The treatment and scaling of weights in multilevel analysis will be specified
separately when the multilevel modelling stage is implemented.

---

### Ukraine

In ESS Round 11 Ukraine, `dweight` and `pspwght` are identical in the analytical
data.

The standard ESS analysis-weight relationship remains valid:

`anweight = pspwght × pweight`

Ukraine therefore remains in the primary V1 analytical sample.

---

## Română

### Scop

Datele ESS la nivel de respondent provin din eșantioane probabilistice complexe.
Din acest motiv, statisticile neponderate ale eșantionului nu sunt tratate în
acest proiect drept estimări finale ale populației.

Proiectul păstrează variabilele ESS originale pentru ponderare și designul
eșantionului:

- `dweight`
- `pspwght`
- `pweight`
- `anweight`
- `psu`
- `stratum`

Nicio pondere ESS nu este modificată manual.

---

### Validarea ponderilor

Pentru eșantionul analitic V1 din ESS Round 11:

- sunt incluși 49.210 respondenți;
- sunt incluse 29 de țări;
- toate cele patru variabile de ponderare sunt disponibile pentru fiecare
  respondent;
- nu există ponderi zero sau negative;
- relația ESS pentru analysis weight a fost confirmată empiric:

`anweight = pspwght × pweight`

cu o diferență numerică maximă observată de:

`0.000000000000`

Population weight (`pweight`) este constant în interiorul fiecărei țări.

Prin urmare, în interiorul unei singure țări, estimările punctuale ponderate cu
`pspwght` și `anweight` sunt numeric echivalente.

Această echivalență a fost verificată direct pentru principalele variabile
analitice.

---

### Ponderea principală în V1

Pentru estimările descriptive punctuale folosim:

`anweight`

Astfel păstrăm o regulă unitară de ponderare în cadrul proiectului.

---

### Estimări la nivel de țară

Mediile și proporțiile la nivel de țară folosesc:

`anweight`

Exemple:

- media auto-poziționării Left-Right;
- proporțiile Left / Centre / Right;
- media Social Trust;
- mediile scorurilor centrate Human Values.

Deoarece `pweight` este constant în interiorul unei țări, aceste estimări sunt
echivalente cu cele obținute folosind `pspwght`.

---

### Estimări regionale și pooled

V1 nu raportează estimări regionale sau estimări pooled la nivel de respondent,
obținute prin combinarea mai multor țări.

Pentru analize viitoare care combină respondenți din mai multe țări, `anweight`
este o posibilă pondere atunci când estimandul urmărit presupune contribuții
proporționale cu dimensiunea populației fiecărei țări. Alegerea finală a
ponderilor trebuie justificată în funcție de populația-țintă și întrebarea
de cercetare.

Spre deosebire de estimările realizate separat în interiorul fiecărei țări,
componenta legată de mărimea populației contează atunci când obiectivul este
estimarea caracteristicilor unei populații combinate.

Utilizarea corespunzătoare a componentei de populație evită tratarea țărilor
cu eșantioane ESS similare, dar populații foarte diferite, ca și cum ar
contribui egal la o estimare la nivelul populației combinate.

---

### Confidence intervals și standard errors

Ponderea utilizată pentru estimările punctuale nu este suficientă pentru
estimarea incertitudinii în cazul unui design de sondaj complex.

În V1 au fost calculate erori standard și intervale de încredere pentru
anumite estimări ESS agregate la nivel de țară.

Procedura de estimare a incertitudinii încorporează informații relevante
despre designul eșantionării ESS, inclusiv:

- `psu`
- `stratum`
- `anweight`

Aceste estimări ale incertitudinii la nivelul sondajului sunt distincte
de incertitudinea coeficienților Pearson și Spearman calculați între țări.

Incertitudinea de eșantionare asociată estimărilor ESS la nivel de țară
nu este propagată integral în cele 36 de corelații dintre țări.

Prin urmare, corelațiile raportate nu trebuie interpretate ca incluzând
toate sursele de incertitudine asociate estimărilor ESS subiacente.

---

### Datasetul analitic la nivel de țară

După agregarea respondent-level, proiectul obține un dataset cu o singură
observație pentru fiecare țară.

Din momentul în care unitatea de analiză devine țara, ponderile respondenților
nu sunt reutilizate automat în modelele country-level.

De exemplu:

`GDP PPS ↔ Mean Social Trust`

reprezintă o relație între țări.

În cele 36 de corelații Pearson și Spearman la nivel de țară din V1, fiecare
țară pentru care există date în perechea analizată contribuie egal. Ponderile
ESS sunt folosite pentru estimarea agregatelor naționale, dar nu sunt aplicate
din nou la calcularea corelațiilor dintre țări.

Prin urmare, politica de ponderare pentru alte modele country-level trebuie să
fie stabilită în funcție de estimandul de la nivelul țării și nu moștenită
mecanic din microdate.

---

### Modele multinivel

Ponderile survey nu vor fi introduse mecanic într-un model multinivel.

Utilizarea și scalarea ponderilor pentru multilevel analysis vor fi stabilite
separat în etapa dedicată modelelor multinivel.

---

### Ucraina

În datele analitice ESS Round 11 pentru Ucraina, `dweight` și `pspwght` sunt
identice.

Relația standard ESS pentru analysis weight rămâne validă:

`anweight = pspwght × pweight`

Prin urmare, Ucraina rămâne în eșantionul analitic principal V1.
