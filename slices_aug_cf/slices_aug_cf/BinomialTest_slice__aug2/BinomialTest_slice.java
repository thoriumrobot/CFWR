/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static long binomial(
      @NonNegative @LTLengthOf("BinomialTest.factorials") int n,
      @NonNegative @LessThan("#1 + 1") int k) {
        try {
            try {
            for (int __cfwr_i72 = 0; __cfwr_i72 < 3; __cfwr_i72++) {
            if (false && (null - ('g' & -988))) {
            while ((null + -928L)) {
            try {
            try {
            int __cfwr_elem74 = -422;
        } catch (Exception __cfwr_e88) {
            // ignore
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        } catch (Exception __cfwr_e58) {
            // ignore
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
      private Double __cfwr_temp970(double __cfwr_p0, short __cfwr_p1) {
        return null;
        return null;
        return null;
    }
}
