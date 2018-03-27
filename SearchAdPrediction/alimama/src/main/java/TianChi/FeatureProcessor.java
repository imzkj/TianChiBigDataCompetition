package TianChi;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.feature.StandardScaler;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.hive.HiveContext;


import scala.Tuple2;

//spark-submit --queue root.bd_app.fdbd_rec --driver-memory 8G --executor-memory 12G --master yarn
// --class com.feidee.classifaction.business.relation_prediction.FeatureProcessor  fd-recommend-offline-0.1.4-jar-with-dependencies.jar
public class FeatureProcessor implements Serializable {

	/**
	 *
	 */
	private static final long serialVersionUID = -8152429227009420462L;
	private static final String serialFilePath = "target/tmp/ijcai_scaler.obj";

	public static void main(String args[]) throws IOException {
		SparkConf sparkConf = new SparkConf();
		JavaSparkContext jsc = new JavaSparkContext(sparkConf);
		HiveContext hiveCtx = new HiveContext(jsc);

		FeatureProcessor processor = new FeatureProcessor();

		JavaRDD<Map<String, Object>> srcData = processor.getTrainData("target/tmp/file/round1_ijcai_18_train_20180301.txt", jsc);
		HbaseManager.saveResultToHbase(srcData, "acc_ijcai_18_train");

		JavaRDD<LabeledPoint> trainRdd = generateTrainData(hiveCtx);

		processor.evalute(jsc, trainRdd, "", "", 2, 0.4);
	}

	public void evalute(JavaSparkContext jsc, JavaRDD<LabeledPoint> trainData,
						String isSaveModel, String modelPath, int classnum, double l2) {
		JavaRDD<LabeledPoint>[] splits = trainData.randomSplit(new double[]{0.7, 0.3}, 11L);
		JavaRDD<LabeledPoint> training = splits[0].cache();
		JavaRDD<LabeledPoint> test = splits[1];
		//test = getOneCredits(test);
		System.out.println("测试集大小：" + test.count());
		LogisticRegressionWithLBFGS lr = new LogisticRegressionWithLBFGS();
		lr.optimizer().setRegParam(l2);
		final LogisticRegressionModel model = lr.setNumClasses(classnum).run(training.rdd());
		//model.clearThreshold();

		JavaRDD<Tuple2<Object, Object>> predictionAndLabels = test.map(
				new Function<LabeledPoint, Tuple2<Object, Object>>() {
					/**
					 *
					 */
					private static final long serialVersionUID = -4527870883965427708L;

					public Tuple2<Object, Object> call(LabeledPoint p) {
						Double prediction = model.predict(p.features());

						return new Tuple2<Object, Object>(prediction, p.label());
					}
				}
		);

		// Get evaluation metrics.
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());

		// Confusion matrix
		Matrix confusion = metrics.confusionMatrix();
		System.out.println("Confusion matrix: \n" + confusion);

		double recall = metrics.recall();
		double precision = metrics.precision();
		System.out.println("precision = " + precision);
		System.out.println("recall = " + recall);
		System.out.println("fMeasure = " + metrics.fMeasure());

		// Stats by labels
		for (int i = 0; i < metrics.labels().length; i++) {
			System.out.format("Class %f precision = %f\n", metrics.labels()[i], metrics.precision
					(metrics.labels()[i]));
			System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(metrics
					.labels()[i]));
			System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure
					(metrics.labels()[i]));
		}

		//Weighted stats
		System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
		System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
		System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
		System.out.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate());


	}

	private JavaRDD<Map<String, Object>> getTrainData(String fileName, JavaSparkContext jsc) {
		JavaRDD<String> lines = jsc.textFile(fileName);

		return lines.map(new Function<String, Map<String, Object>>() {

			/**
			 *
			 */
			private static final long serialVersionUID = -5881321144244171135L;

			@Override
			public Map<String, Object> call(String line) {
				Map<String, Object> result = new HashMap<>();

				String[] fields = line.split(" ");
				String instanceId = fields[0];
				result.put("id", instanceId);
				result.put("instanceid", instanceId);

				String itemId = fields[1];
				result.put("itemid", itemId);
				String itemcategorylist = fields[2];
				result.put("itemcategorylist", itemcategorylist);
				String itempropertylist = fields[3];
				result.put("itempropertylist", itempropertylist);
				String item_brand_id = fields[4];
				result.put("itembrandid", item_brand_id);
				String itemcityid = fields[5];
				result.put("itemcityid", itemcityid);
				String itempricelevel = fields[6];
				result.put("itempricelevel", itempricelevel);
				String item_saleslevel = fields[7];
				result.put("itemsaleslevel", item_saleslevel);
				String itemcollectedlevel = fields[8];
				result.put("itemcollectedlevel", itemcollectedlevel);
				String itempvlevel = fields[9];
				result.put("itempvlevel", itempvlevel);
				String userid = fields[10];
				result.put("userid", userid);
				String usergenderid = fields[11];
				result.put("usergenderid", usergenderid);
				String useragelevel = fields[12];
				result.put("useragelevel", useragelevel);
				String useroccupationid = fields[13];
				result.put("useroccupationid", useroccupationid);
				String userstarlevel = fields[14];
				result.put("userstarlevel", userstarlevel);
				String contextid = fields[15];
				result.put("contextid", contextid);
				String contexttimestamp = fields[16];
				result.put("contexttimestamp", contexttimestamp);
				String context_page_id = fields[17];
				result.put("context_page_id", context_page_id);
				String predict_category_property = fields[18];
				result.put("predict_category_property", predict_category_property);
				String shop_id = fields[19];
				result.put("shopid", shop_id);
				String shop_review_num_level = fields[20];
				result.put("shop_review_num_level", shop_review_num_level);
				String shop_review_positive_rate = fields[21];
				result.put("shop_review_positive_rate", shop_review_positive_rate);
				String shop_star_level = fields[22];
				result.put("shop_star_level", shop_star_level);
				String shop_score_service = fields[23];
				result.put("shop_score_service", shop_score_service);
				String shop_score_delivery = fields[24];
				result.put("shop_score_delivery", shop_score_delivery);
				String shop_score_description = fields[25];
				result.put("shop_score_description", shop_score_description);
				String is_trade = fields[26];
				result.put("is_trade", is_trade);
				return result;
			}
		});


	}

	public static JavaRDD<LabeledPoint> generateTrainData(HiveContext hiveCtx) {
		String sqlText = "SELECT itempricelevel, itemsaleslevel, itemcollectedlevel,"
				+ "itempvlevel,usergenderid, useragelevel, useroccupationid,"
				+ "userstarlevel, contexttimestamp, context_page_id,"
				+ "shop_review_num_level, shop_review_positive_rate, shop_star_level,"
				+ "shop_score_service, shop_score_delivery, shop_score_description,"
				+ "is_trade from temp.acc_ijcai_18_train WHERE itemsaleslevel!='-1' "
				+ "AND usergenderid!='-1' AND useragelevel != '-1' AND useroccupationid!='-1'";

		JavaRDD<Row> srcDataRdd = hiveCtx.sql(sqlText).toJavaRDD();
		srcDataRdd.cache();
		JavaRDD<Vector> featureRdd = srcDataRdd.map(new Function<Row, Vector>() {

			/**
			 *
			 */
			private static final long serialVersionUID = 4561635363235393999L;

			@Override
			public Vector call(Row row) {
				List<Double> features = new ArrayList<>();

				double item_price_level = Double.parseDouble(row.getString(0));
				features.add(item_price_level);
				double item_sales_level = Double.parseDouble(row.getString(1));
				features.add(item_sales_level);
				double item_collected_level = Double.parseDouble(row.getString(2));
				features.add(item_collected_level);
				double item_pv_level = Double.parseDouble(row.getString(3));
				features.add(item_pv_level);
				String user_gender_id = row.getString(4);
				double[] usergenderidCode = genderOneHot(user_gender_id);
				for (double val : usergenderidCode) {
					features.add(val);
				}
				double user_age_level = Double.parseDouble(row.getString(5));
				features.add(user_age_level);
				String user_occupation_id = row.getString(6);
				double[] occupationidOneHotCode = occupationidOneHot(user_occupation_id);
				for (double val : occupationidOneHotCode) {
					features.add(val);
				}
				double user_star_level = Double.parseDouble(row.getString(7));
				features.add(user_star_level);
				double context_time_stamp = Double.parseDouble(row.getString(8));
				features.add(context_time_stamp);
				double context_page_id = Double.parseDouble(row.getString(9));
				features.add(context_page_id);
				double shop_review_num_level = Double.parseDouble(row.getString(10));
				features.add(shop_review_num_level);
				double shop_review_positive_rate = Double.parseDouble(row.getString(11));
				features.add(shop_review_positive_rate);

				double shop_star_level = Double.parseDouble(row.getString(12));
				features.add(shop_star_level);
				double shop_score_service = Double.parseDouble(row.getString(13));
				features.add(shop_score_service);
				double shop_score_delivery = Double.parseDouble(row.getString(14));
				features.add(shop_score_delivery);
				double shop_score_description = Double.parseDouble(row.getString(15));
				features.add(shop_score_description);

				double[] vector = new double[features.size()];
				for (int i = 0; i < vector.length; i++) {
					vector[i] = features.get(i);
				}

				return Vectors.dense(vector);
			}
		});

		StandardScaler scaler = new StandardScaler(true, true);
		final StandardScalerModel scalerModel = scaler.fit(featureRdd.rdd());

		//序列化scalerModel
		Tools.serializeObj(serialFilePath, scalerModel);

		return srcDataRdd.map(new Function<Row, LabeledPoint>() {

			/**
			 *
			 */
			private static final long serialVersionUID = 5480527922274966355L;

			@Override
			public LabeledPoint call(Row row) {
				List<Double> features = new ArrayList<>();

				double item_price_level = Double.parseDouble(row.getString(0));
				features.add(item_price_level);
				double item_sales_level = Double.parseDouble(row.getString(1));
				features.add(item_sales_level);
				double item_collected_level = Double.parseDouble(row.getString(2));
				features.add(item_collected_level);
				double item_pv_level = Double.parseDouble(row.getString(3));
				features.add(item_pv_level);
				String user_gender_id = row.getString(4);
				double[] usergenderidCode = genderOneHot(user_gender_id);
				for (double val : usergenderidCode) {
					features.add(val);
				}
				double user_age_level = Double.parseDouble(row.getString(5));
				features.add(user_age_level);
				String user_occupation_id = row.getString(6);
				double[] occupationidOneHotCode = occupationidOneHot(user_occupation_id);
				for (double val : occupationidOneHotCode) {
					features.add(val);
				}
				double user_star_level = Double.parseDouble(row.getString(7));
				features.add(user_star_level);
				double context_time_stamp = Double.parseDouble(row.getString(8));
				features.add(context_time_stamp);
				double context_page_id = Double.parseDouble(row.getString(9));
				features.add(context_page_id);
				double shop_review_num_level = Double.parseDouble(row.getString(10));
				features.add(shop_review_num_level);
				double shop_review_positive_rate = Double.parseDouble(row.getString(11));
				features.add(shop_review_positive_rate);

				double shop_star_level = Double.parseDouble(row.getString(12));
				features.add(shop_star_level);
				double shop_score_service = Double.parseDouble(row.getString(13));
				features.add(shop_score_service);
				double shop_score_delivery = Double.parseDouble(row.getString(14));
				features.add(shop_score_delivery);
				double shop_score_description = Double.parseDouble(row.getString(15));
				features.add(shop_score_description);

				double[] vector = new double[features.size()];
				for (int i = 0; i < vector.length; i++) {
					vector[i] = features.get(i);
				}

				double is_trade = Double.parseDouble(row.getString(16));

				Vector feat_vec = scalerModel.transform(Vectors.dense(vector));

				return new LabeledPoint(is_trade, feat_vec);
			}
		});
	}

	//性别临时one-hot编码方法
	private static double[] genderOneHot(String user_gender_id) {
		double[] genderCode = new double[]{0, 0, 0};
		genderCode[Integer.parseInt(user_gender_id)] = 1;
		return genderCode;

	}

	//职业临时one-hot编码方法
	private static double[] occupationidOneHot(String user_occupation_id) {
		Map<String, Integer> occupationidMap = new HashMap<>();
		occupationidMap.put("2002", 0);
		occupationidMap.put("2003", 1);
		occupationidMap.put("2004", 2);
		occupationidMap.put("2005", 3);

		double[] occupationidCode = new double[]{0, 0, 0, 0};
		occupationidCode[occupationidMap.get(user_occupation_id)] = 1;
		return occupationidCode;
	}
}
