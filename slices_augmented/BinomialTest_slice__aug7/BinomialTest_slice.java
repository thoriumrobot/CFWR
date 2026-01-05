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
        long __cfwr_item56 = 200L;

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

    public String __cfwr_process782(Double __cfwr_p0) {
        if ((null >> 938L) && false) {
            if (true && false) {
            String __cfwr_obj13 = "value5";
        }
        }
        if (false && true) {
            return null;
        }
        return "test12";
    }
    protected static Boolean __cfwr_util24() {
        for (int __cfwr_i37 = 0; __cfwr_i37 < 1; __cfwr_i37++) {
            if (false && true) {
            Object __cfwr_node71 = null;
        }
        }
        return null;
    }
}