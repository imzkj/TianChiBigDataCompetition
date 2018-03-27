package TianChi;

import java.io.IOException;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.TableOutputFormat;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.mapreduce.Job;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

public class HbaseManager {

	public static void saveResultToHbase(JavaRDD<Map<String, Object>> rdd, String resultTable) throws IOException {
		JavaPairRDD<ImmutableBytesWritable, Put> hbaseData = rdd.mapToPair(new PairFunction<Map<String, Object>, ImmutableBytesWritable, Put>() {

			private static final long serialVersionUID = 1L;

			@Override
			public Tuple2<ImmutableBytesWritable, Put> call(Map<String, Object> user) throws Exception {
				String id = String.valueOf(user.get("id"));

				Put put = new Put(Bytes.toBytes(Tools.string2MD5(id)));
				//put.addColumn(Bytes.toBytes("info"), Bytes.toBytes("sid"), Bytes.toBytes(sid));

				for (String key : user.keySet()) {
					put.addColumn(Bytes.toBytes("info"), Bytes.toBytes(key), Bytes.toBytes(String.valueOf(user.get(key))));
				}


				return new Tuple2<>(new ImmutableBytesWritable(), put);
			}
		});

		Configuration configuration = HBaseConfiguration.create();
		configuration.set("hbase.zookeeper.quorum", "hbase.zookeeper.quorum");
		configuration.set("hbase.zookeeper.property.clientPort", "hbase.zookeeper.property.clientPort");

		Job job = Job.getInstance(configuration);
		job.setOutputFormatClass(TableOutputFormat.class);
		job.getConfiguration().set(TableOutputFormat.OUTPUT_TABLE, resultTable);
		hbaseData.saveAsNewAPIHadoopDataset(job.getConfiguration());
	}
}
