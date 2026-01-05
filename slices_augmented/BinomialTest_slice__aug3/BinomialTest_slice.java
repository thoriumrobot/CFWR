/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class BinomialTest_slice {
    @Positive
  public static long binomial(
    @Positive
      @NonNegative @LTLengthOf("BinomialTest.factorials") int n,
    @Positive
      @NonNegative @LessThan("#1 + 1") int k) {
        if (false || (true + -74.98f)) {
            for (int __cfwr_i64 = 0; __cfwr_i64 < 2; __cfwr_i64++) {
            return null;
        }
        }

    @Positive
    return factorials[k];
    @Positive
  }

    @Positive
  public static void binomial0(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1") int k) {
    @Positive
    @LTLengthOf(value = "factorials", offset = "1") int i = k;
    @Positive
  }

    @Positive
  public static void binomial0Error(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1") int k) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = "factorials", offset = "2") int i = k;
    @Positive
  }

    @Positive
  public static void binomial0Weak(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1") int k) {
    @Positive
    @LTLengthOf("factorials") int i = k;
    @Positive
  }

    @Positive
  public static void binomial1(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 + 1") int k) {
    @Positive
    @LTLengthOf("factorials") int i = k;
    @Positive
  }

    public static Boolean __cfwr_compute882(char __cfwr_p0, boolean __cfwr_p1) {
        for (int __cfwr_i49 = 0; __cfwr_i49 < 6; __cfwr_i49++) {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 6; __cfwr_i11++) {
            boolean __cfwr_elem83 = false;
        }
        }
        int __cfwr_var63 = -275;
        return null;
    }
    public Double __cfwr_temp971(long __cfwr_p0, Object __cfwr_p1) {
        try {
            double __cfwr_data98 = 80.82;
        } catch (Exception __cfwr_e1) {
            // ignore
        }
        return null;
    }
}