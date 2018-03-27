package TianChi;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Tools {

	private static Logger logger = LoggerFactory.getLogger(Tools.class);

	/**
	 * 序列化对象
	 */
	public static void serializeObj(String saveObjFile, Object saveObject) {
		try {
			//初始时staticVar为5
			ObjectOutputStream out = new ObjectOutputStream(
					new FileOutputStream(saveObjFile));
			out.writeObject(saveObject);
			out.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * 反序列化对象
	 */
	public static Object deserializeObj(String saveObjFile) {

		ObjectInputStream oin;
		Object obj = null;
		try {
			oin = new ObjectInputStream(new FileInputStream(
					saveObjFile));
			obj = oin.readObject();
			oin.close();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return obj;
	}

	public static Map<String, String> parseArgs(String[] args) {
		Map<String, String> result = new HashMap<>();
		if (args == null || args.length == 0) {
			return result;
		}

		for (String arg : args) {
			String[] kv = arg.split("=");
			result.put(kv[0], kv[1]);
		}

		return result;
	}

	/**
	 * 余弦相似度
	 *
	 * @param vector1
	 * @param vector2
	 * @return
	 */
	public static double consine(double[] vector1, double[] vector2) {
		if (vector1 == null || vector2 == null) {
			logger.error("vector1 : {} or vector2 : {} is null", vector1, vector2);
			throw new NullPointerException();
		}

		if (vector1.length != vector2.length || vector1.length == 0) {
			logger.error("vector1'length is not equal to vector2.length");
			throw new IllegalArgumentException();
		}
		double dotProduct = 0.0;
		for (int i = 0; i < vector1.length; i++) {
			dotProduct += vector1[i] * vector2[i];
		}
		return dotProduct / (mode(vector1) * mode(vector2));
	}

	/**
	 * @param vector
	 * @return vector的模
	 */
	public static double mode(double[] vector) {
		if (vector == null) {
			throw new NullPointerException();
		}

		if (vector.length == 0) {
			throw new IllegalArgumentException();
		}

		double result = 0.0;
		for (double d : vector) {
			result += d * d;
		}
		return Math.sqrt(result);
	}

	/***
	 * MD5加码 生成32位md5码
	 */
	public static String string2MD5(String inStr) {
		MessageDigest md5 = null;
		try {
			md5 = MessageDigest.getInstance("MD5");
		} catch (Exception e) {
			System.out.println(e.toString());
			e.printStackTrace();
			return "";
		}
		char[] charArray = inStr.toCharArray();
		byte[] byteArray = new byte[charArray.length];

		for (int i = 0; i < charArray.length; i++)
			byteArray[i] = (byte) charArray[i];
		byte[] md5Bytes = md5.digest(byteArray);
		StringBuffer hexValue = new StringBuffer();
		for (int i = 0; i < md5Bytes.length; i++) {
			int val = ((int) md5Bytes[i]) & 0xff;
			if (val < 16)
				hexValue.append("0");
			hexValue.append(Integer.toHexString(val));
		}
		return hexValue.toString();

	}

	/**
	 * 加密解密算法 执行一次加密，两次解密
	 */
	public static String convertMD5(String inStr) {

		char[] a = inStr.toCharArray();
		for (int i = 0; i < a.length; i++) {
			a[i] = (char) (a[i] ^ 't');
		}
		return new String(a);
	}
}
