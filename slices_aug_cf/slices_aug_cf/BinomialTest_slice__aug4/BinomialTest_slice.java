/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static long binomial(
      @NonNegative @LTLengthOf("BinomialTest.factorials") int n,
      @NonNegative @LessThan("#1 + 1") int k) {
        try {
            return 'U';
        } catch (Exception __cfwr_e87) {
            // ignore
        }

    return factorials[k];
 
        try {
            Boolean __cfwr_item65 = null;
        } catch (Exception __cfwr_e23) {
            // ignore
        }
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
      private static int __cfwr_handle20(Double __cfwr_p0, byte __cfwr_p1) {
        while (false) {
            return null;
            break; // Prevent infinite loops
        }
        try {
            while (false) {
            if (false && false) {
            Boolean __cfwr_data86 = null;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e66) {
            // ignore
        }
        int __cfwr_item98 = ((-98.07f >> 652L) - '6');
        try {
            boolean __cfwr_entry81 = false;
        } catch (Exception __cfwr_e81) {
            // ignore
        }
        return 609;
    }
    private static long __cfwr_temp205() {
        double __cfwr_val56 = -53.38;
        double __cfwr_var42 = -61.47;
        return 450L;
    }
}
