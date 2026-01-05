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
        Long __cfwr_entry51 = null;

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

    public static Float __cfwr_calc595(float __cfwr_p0, Float __cfwr_p1) {
        short __cfwr_data46 = null;
        Integer __cfwr_elem26 = null;
        return null;
    }
}