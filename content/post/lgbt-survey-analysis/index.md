---
title: "LGBT Survey Analysis"
date: "2019-07-20"
summary: Analysis of a survey among LGBT representatives in EU in 2012.
image:
  caption: 'Image by <a href="https://pixabay.com/users/Wokandapix-614097/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=2495948">Wokandapix</a> from <a href="https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=2495948">Pixabay</a>'
  focal_point: ""
  placement: 3
  preview_only: false
categories: ["Data Visualization"]
tags: ["Data Visualization", "Survey Analysis", "Python"]
---

**Table of Contents**
- <a href='#intro'>1. Project overview and objectives</a> 
    - <a href='#survey'>1.1. The aim of the survey</a>
    - <a href='#data'>1.2. Data set overview</a>
- <a href='#bi'>2. Choropleth map visualization of responses</a>
- <a href='#score'>3. Country 'suitable' scores</a>
    - <a href='#method'>3.1. Scoring methodology</a>
    - <a href='#dl'>3.2. Daily Life</a>
    - <a href='#ra'>3.3. Right Awareness</a>
    - <a href='#disc'>3.4. Discrimination</a>
    - <a href='#vah'>3.5. Violence and Harassment</a>
    - <a href='#overall'>3.6. Overall rank</a>
- <a href='#lbgt'>4. What the LGBT community says</a>
    - <a href='#satisfied'>4.1. Do people fell satisfied in EU countries?</a>
    - <a href='#open'>4.2. Are people being open about their orientation?</a>
    - <a href='#comf'>4.3. What would allow to live more comfortable?</a>
- <a href='#end'>5. Conclusions</a>

**Note: I've hidden all the code blocks since they have taken so much place and have been somewhat distracting. I pushed the [notebook](https://nbviewer.jupyter.org/github/ruslan-kl/lgbt/blob/master/lgbt-survey-analysis.ipynb) to [GitHub repo](https://github.com/ruslan-kl/lgbt).**

# <a id='intro'>1. Project overview and objectives</a>

The main purpose of this project is the visualization of survey results conducted in EU countries (and Croatia) among 93000 LGBT people (2012). I tried to estimate the overall score of "suitability" (in other words, how good is this county for LGBT community?) by assigning weights to answers and getting average scores for each of the question block. Then I look at some particular questions to explore how satisfied LGBT communiy is and what they think would improve their lives in the countries they live in.

## <a id='survey'>1.1. The aim of the survey</a>

> *The aim of the EU LGBT survey was to obtain robust and comparable data that would allow a better understanding of how lesbian, gay, bisexual and transgender (LGBT) people living in the European Union (EU) and Croatia experience the enjoyment of fundamental rights. The survey collected data from 93,079 people across the EU and Croatia through an anonymous online questionnaire, collecting the views, perceptions, opinions and experiences of persons aged 18 years or over, who self-identify as lesbian, gay, bisexual or transgender. The topics related to various fundamental rights issues with an emphasis on experienced discrimination, violence and harassment. The survey and all related activities covered the 27 current EU Member States as well as Croatia. FRA designed the questionnaire and finalised it in consultation with its Scientific Committee, relevant stakeholders and civil society organisations, as well as independent academics and national experts with expertise in the area of discrimination on grounds of sexual orientation and
gender identity.*
>
> *The survey asked a range of questions about LGBT people’s experiences including:*
> * *public perceptions and responses to homophobia and/or transphobia;*
> * *discrimination;*
> * *rights awareness;*
> * *safe environment;*
> * *violence and harassment;*
> * *the social context of being an LGBT person;*
> * *personal characteristics, including age and income group.*

*Taken from [EU LGBT survey technical report. Methodology, online survey, questionnaire and sample](https://fra.europa.eu/sites/default/files/eu-lgbt-survey-technical-report_en.pdf)*

## <a id='data'>1.2. Data set overview</a>

Data set consist of 5 .csv files that represent 5 blocks of questions.

The schema of all the tables is identical:

| Variable | Note/Example |
|:-:|:-:|
| `CountryCode` | name of the country |
| `subset` | `Lesbian`, `Gay`, `Bisexual women`, `Bisexual men` or `Transgender` |
| `question_code` | unique code ID for the question |
| `question_label` | full question text |
| `answer` | answer given |
| `percentage` | % |
| `notes` | `[0]`: small sample size; `[1]`: NA due to small sample size; `[2]`: missing value |


* Total amount of countries that participated in the survey is 28
* All answers are different (i.e, can be binary (`Yes-No`), numerical (`1-10`) or scale (`Always-Often-Never`))

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total number of questions</th>
      <th>Number of records with small sample size</th>
      <th>% of total records(0)</th>
      <th>Number of missing values due to the small sample size</th>
      <th>% of total records(1)</th>
      <th>Number of missing values</th>
      <th>% of total records(2)</th>
    </tr>
    <tr>
      <th>Data set</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Daily Life</th>
      <td>50.0</td>
      <td>13447.0</td>
      <td>39.5</td>
      <td>1849.0</td>
      <td>5.4</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Rights Awareness</th>
      <td>10.0</td>
      <td>785.0</td>
      <td>20.8</td>
      <td>130.0</td>
      <td>3.4</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Violence and Harassment</th>
      <td>47.0</td>
      <td>25072.0</td>
      <td>55.3</td>
      <td>6897.0</td>
      <td>15.2</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Discrimination</th>
      <td>32.0</td>
      <td>5782.0</td>
      <td>36.7</td>
      <td>1198.0</td>
      <td>7.6</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


# <a id='bi'>2. Choropleth map visualization of responses</a>

This visualization allows to explore single question response by country. The dashboard was done using **Microsoft Power BI**. Original map visualization can be find [here](https://fra.europa.eu/en/publications-and-resources/data-and-maps/survey-fundamental-rights-lesbian-gay-bisexual-and).

<iframe width="700" height="800" src="https://app.powerbi.com/view?r=eyJrIjoiMzI4MzMzN2QtYTA5NC00MTZkLTllYTAtMWMzOWQxNjlmZjI5IiwidCI6ImMzNWFiZTIwLTI1N2QtNDcxZi04ZDI3LWU3MTI5ZjA5MjJmNSIsImMiOjl9" frameborder="0" allowFullScreen="true"></iframe>

# <a id='score'>3. Country 'suitable' scores</a>

In this section I am going to score each country by the survey answers to find out which county is "most suitable" for LGBT community. Each country will get a score in 4 blocks **Daily Life**, **Discrimination**, **Violence and Harassment** and **Rights Awareness** (I didn't include **Transgender Specific Questions** here since the segment of people is transgenders only) and a **final score**. 

## <a id='method'>3.1. Scoring methodology</a>

First of all the ratio of Lesbians/Gays/Bisexuals/Transgenders are not equal amoung countries. In order to 'normalize' I am going to set the weight of each subset:

\begin{align*}
\textrm{Weight}_{\text{subset}} = \frac{\text{# Subset for a Country}}{\text{# Total for a Country}}
\end{align*}


Final `Subset Weight` values look like this:


<iframe src="plotly-output/1.html" width="800" height="450" frameborder="0"></iframe>


After calculating the `Subset Weight` values I am going to get new value of `Percent` of responses for each subset by multiplying the original `Percent` value by the `Subset Weight`.

\begin{align*}
\textrm{Percent}_{\textrm{weighted}} = \textrm{Percent} \times \textrm{Weight}_{\textrm{subset}}
\end{align*}

After this I am adding a `Response Weight` which will show how 'good' the answer is. Let's take a look at imaginary example for two qestions for `Italy`:

| Country | Question | Answer | Percent (Weighted) |
|:-:|:-:|:-:|:-:|
| Italy | In your opinion, in the country where you live, how widespread is discrimination because a person is Transgender? | Very widespread | 25 |
| Italy | In your opinion, in the country where you live, how widespread is discrimination because a person is Transgender? | Fairly widespread | 15 |
| Italy | In your opinion, in the country where you live, how widespread is discrimination because a person is Transgender? | Dont know | 10 |
| Italy | In your opinion, in the country where you live, how widespread is discrimination because a person is Transgender? | Fairly rare | 30 |
| Italy | In your opinion, in the country where you live, how widespread is discrimination because a person is Transgender? | Very rare | 20 |
| Italy | Have you personally felt discriminated against or harassed because of being perceived as Gay?	| Yes | 30 |
| Italy | Have you personally felt discriminated against or harassed because of being perceived as Gay?	| Don't know | 20 |
| Italy | Have you personally felt discriminated against or harassed because of being perceived as Gay?	| No | 50 |



First step is going to be adding a weight to each answer in range `[-1, 1]` with `-1` being negative and `1` being positive. Looking at the first question `In your opinion, in the country where you live, how widespread is discrimination because a person is Transgender?` the answer `Very rare` is the best possible scenario among all the answer options while `Very widespread` is the worst. So I'm assigning weight `-1` to `Very widespread` and `1` to `Very rare`. The rest weight of the answers are splited evenly (`-0.5` to `Fairly widespread` and `0.5` to `Fairly rare`). For example if there is 6 answer options, the weights look like this `[-1, -0.66, -0.33, 0.33, 0.66, 1]`. Answer option `Don't know` gets `np.NaN`. 

*Note: before I thought that `Don't know` answer weight should be `0` but then I changed it to `np.NaN` so it doesn't affect the total score since that answer is not really helpful. If you think it should be `0` I would love to hear your reasons.*

Then I compute the `Score` by following formula:

\begin{align*}
\textrm{Score} = \textrm{Weight}_{\textrm{response}} \times \frac{\textrm{Percent}_{\textrm{weighted}} }{100}
\end{align*}

In that case `Score` can also be in the range `[-1, 1]` with `-1` being negative and `1` being positive. The final `Total Block Score` for the country is just taking the average of all the scores.

\begin{align*}
\textrm{Total Block Score} = \textrm{Average(Score)}
\end{align*}

| Country | Question | Answer | Percent (Weighted) | Response Weight | Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| Italy | In your opinion, in the country where you live, how widespread is discrimination because a person is Transgender? | Very widespread | 25 | -1 | -0.25 | 
| Italy | In your opinion, in the country where you live, how widespread is discrimination because a person is Transgender? | Fairly widespread | 15 | -0.5 | -0.075 |
| Italy | In your opinion, in the country where you live, how widespread is discrimination because a person is Transgender? | Don't know | 10 | np.NaN | np.NaN |
| Italy | In your opinion, in the country where you live, how widespread is discrimination because a person is Transgender? | Fairly rare | 30 | 0.5 | 0.15 |
| Italy | In your opinion, in the country where you live, how widespread is discrimination because a person is Transgender? | Very rare | 20 | 1 | 0.2 |
| Italy | Have you personally felt discriminated against or harassed because of being perceived as Gay?	| Yes | 30 | -1 | -0.3 |
| Italy | Have you personally felt discriminated against or harassed because of being perceived as Gay?	| Don't know | 20 | np.NaN | np.NaN |
| Italy | Have you personally felt discriminated against or harassed because of being perceived as Gay?	| No | 50 | 1 | 0.2 |


<br>
So the `Total Block Score` for this block is going to be $\frac{-0.25 -.075 + 0.15 + 0.2 - 0.3 + 0.2}{6} = −0.0125$. After computing the scores for 4 blocks the `Total Score` is going to be the average of four `Total Block Scores`.

## <a id='dl'>3.2. Daily Life</a>

Let's start with `Daily Life` questions block where subjects answered questions about day to day living as a lesbian, gay, bisexual or transgender person.


<iframe src="plotly-output/2.html" width="700" height="450" frameborder="0"></iframe>

* The first place goes to **Netherlands**🇳🇱 (which means that the responses about daily life for this country were more positive comparing to other countries).
* The last place goes to **Cyprus**🇨🇾.

## <a id='ra'>3.3. Right Awareness</a>


<iframe src="plotly-output/3.html" width="700" height="450" frameborder="0"></iframe>

* The first place goes to **Sweden🇸🇪** (which means that the people from the LGBT community are much more aware about their rights in that country comparing to other).
* The last place goes to **Greece**🇬🇷.

## <a id='disc'>3.4. Discrimination</a>


<iframe src="plotly-output/4.html" width="700" height="450" frameborder="0"></iframe>


* The first place goes to **Malta**🇲🇹 (which means that the people from the LGBT feel less discriminated in that country comparing to other countries).
* The last place goes to **Romania**🇷🇴.

## <a id='vah'>3.5. Violence and Harassment</a>


<iframe src="plotly-output/5.html" width="700" height="450" frameborder="0"></iframe>


* The first place goes to **Finland🇫🇮** (which means that the people from the LGBT are beinge the subject of harassment or violation less often in that country comparing to other countries).
* The last place goes to **Estonia**🇪🇪.

## <a id='overall'>3.6. Overall rank</a>

By taking the average of 4 scores we can rescale that values to get the final `Total Rank`.


<iframe src="plotly-output/6.html" width="700" height="450" frameborder="0"></iframe>


So!

* The absolute winners are **Denmark**🇩🇰, **Netherlands**🇳🇱, **Sweden**🇸🇪.
* The absolute losers are **Romania**🇧🇬, **Bulgaria**🇧🇬, **Cyprus**🇨🇾.

Here is something to think about when you are considering a destination for travelling/relocation.

# <a id='lbgt'>4. What the LGBT community says</a>

After I got the `Total Rank` for each country I want to look at some particular responses to find out how does LGBT community respond to living in EU countries.

## <a id='satisfied'>4.1.  Do people fell satisfied in EU countries?</a>

There was a question "**All things considered, how satisfied would you say you are with your life these days?**" in Daily Life questions block where subjects could pick a value from 0 to 10 (10 being the most satisfied) of how satisfied they feel. Using the same methodology I am going to find a score for this single question and compare it to the `Total Rank` from previous section.


<iframe src="plotly-output/7.html" width="700" height="550" frameborder="0"></iframe>


<iframe src="plotly-output/8.html" width="700" height="450" frameborder="0"></iframe>


\begin{align*}
\textrm{Rank Diff} = \textrm{Satisfaction Rank} - \textrm{Total Rank}
\end{align*}

In such way, `-` sign in `Satisfaction Rank` column means that LGBT community feel more satisfied in that county as I would guess from `Total Rank` value. `+` sign tells the opposite.

## <a id='open'>4.2. Are people being open about their orientation?</a>

Next question "**4 levels of being open about LGBT background**" from Daiy Life questions block allow to see how open the LGBT community is in the country they live in. The possible answers are Never Open, Rarerly Open, Fairly Open, Very Open.


<iframe src="plotly-output/9.html" width="800" height="600" frameborder="0"></iframe>


The countries in the plot are sorted by the `Total Rank` (the top countries have the highest rank, the bottom countries have the lowest rank). You can notice how the 'openess ratio' is correlated with country `Total Rank` - the higher the rank, the higher is the ratio of 'open' people.


<iframe src="plotly-output/10.html" width="700" height="500" frameborder="0"></iframe>


<iframe src="plotly-output/11.html" width="700" height="450" frameborder="0"></iframe>


* Gay Men have the highest `Very Open` rate (23%) while Bisexual Men have the highest `Never Open` rate (75%).
* In total, about 27% of people from LGBT community being open about their orientaion (`Very Open` + `Fairly Open`), especially in Netherlands🇳🇱 (15%).

## <a id='comf'>4.3. What would allow to live more comfortable?</a>

There were a series of questions "**What would allow you to be more comfortable living as a LGB person?**" with 8 different options that allow to explore what is missing in current situation in the country for the LGBT community to feel better.

<br>

<iframe src="plotly-output/12.html" width="900" height="600" frameborder=80"></iframe>

<br>

<iframe src="plotly-output/13.html" width="700" height="450" frameborder="0"></iframe>

<br>

<iframe src="plotly-output/14.html" width="700" height="550" frameborder="0"></iframe>


* High ratio of people (88%) agreed that **Measures implemented at school to respect LGB people** would improve the situation (especially in Italy🇮🇹 with 78%)
* 16% of people feel satisfied with the **The possibility to marry and/or register a partnership** (top countries are: Netherlands🇳🇱, Belgium🇧🇪 and Portugal🇵🇹)
* 9% of people don't think that **The possibility to foster / adopt children** would change a lot.

# <a id='end'>5. Conclusions</a>

So I estimated the country ranks of goodness for LGBT community, showed in what countries people are more open about their orientation and what do people think would make their life better. It's just a small piece of insights that could be extracted from this survey so many more questions can be answered. You can also check the [official report](https://fra.europa.eu/en/publication/2014/eu-lgbt-survey-european-union-lesbian-gay-bisexual-and-transgender-survey-main) with survey analysis results.
