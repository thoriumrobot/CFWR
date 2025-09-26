/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static long binomial(
      @NonNegative @LTLengthOf("BinomialTest.factorials") int n,
      @NonNegative @LessThan("#1 + 1") int k) {
        Double __cfwr_item11 = null;

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
      private float __cfwr_func575(Double __cfwr_p0, Object __cfwr_p1, Character __cfwr_p2) {
        return -915L;
        if (false && true) {
            try {
            long __cfwr_result50 = (73 << null);
        } catch (Exception __cfwr_e66) {
            // ignore
        }
        }
        for (int __cfwr_i88 = 0; __cfwr_i88 < 1; __cfwr_i88++) {
            if (((null / -90.95f) * '5') && true) {
            if (false && true) {
            for (int __cfwr_i52 = 0; __cfwr_i52 < 1; __cfwr_i52++) {
            if ((false / (null % -431)) && false) {
            for (int __cfwr_i77 = 0; __cfwr_i77 < 1; __cfwr_i77++) {
            return null;
        }
        }
        }
        }
        }
        }
        return -74.22f;
    }
    public static float __cfwr_util163(double __cfwr_p0, char __cfwr_p1, Integer __cfwr_p2) {
        return "result37";
        try {
            for (int __cfwr_i37 = 0; __cfwr_i37 < 10; __cfwr_i37++) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 2; __cfwr_i14++) {
            while (true) {
            Character __cfwr_obj17 = null;
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e5) {
            // ignore
        }
        return -78.13f;
    }
}
