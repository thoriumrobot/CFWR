/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static long binomial(
      @NonNegative @LTLengthOf("BinomialTest.factorials") int n,
      @NonNegative @LessThan("#1 + 1") int k) {
        for (int __cfwr_i3 = 0; __cfwr_i3 < 4; __cfwr_i3++) {
            for (int __cfwr_i15 = 0; __cfwr_i15 < 3; __cfwr_i15++) {
            if (((278 * -20.15f) + 339) || false) {
            if (true && false) {
            if (((-7L * -60.43f) + -64.84f) || (null << (371 ^ null))) {
            return null;
        }
        }
        }
        }
        }

    return factorials[k];
  }

  public static void binomial0(
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1") int k) {
    @LTLengthOf(value = "factorials", offset = "1") int i = k;
  }

  public static void binomial0Error(
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1") int k) {
    // :: error: (assignment)
    @LTLengthOf(value = "factorials", offset = "2") int i = k;
  }

  public static void binomial0Weak(
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1") int k) {
    @LTLengthOf("factorials") int i = k;
  }

  public static void binomial1(
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 + 1") int k) {
    @LTLengthOf("factorials") int i = k;
  }

  public static void binomial1Error(
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 + 1") int k) {
    // :: error: (assignment)
    @LTLengthOf(value = "factorials", offset = "1") int i = k;
  }

  public static void binomial2(
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 + 2") int k) {
    @LTLengthOf(value = "factorials", offset = "-1") int i = k;
  }

  public static void binomial2Error(
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 + 2") int k) {
    // :: error: (assignment)
    @LTLengthOf(value = "factorials", offset = "0") int i = k;
  }

  public static void binomial_1(
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 - 1") int k) {
    @LTLengthOf(value = "factorials", offset = "2") int i = k;
  }

  public static void binomial_1Error(
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 - 1") int k) {
    // :: error: (assignment)
    @LTLengthOf(value = "factorials", offset = "3") int i = k;
  }

  public static void binomial_2(
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 - 2") int k) {
    @LTLengthOf(value = "factorials", offset = "3") int i = k;
  }

  public static void binomial_2Error(
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 - 2") int k) {
    // :: error: (assignment)
    @LTLengthOf(value = "factorials", offset = "4") int i = k;
      public static Float __cfwr_func157() {
        return null;
        return null;
    }
    private static long __cfwr_aux573(Character __cfwr_p0) {
        short __cfwr_result17 = null;
        return -519L;
    }
}
