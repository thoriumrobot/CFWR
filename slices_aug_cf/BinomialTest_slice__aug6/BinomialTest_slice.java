/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static long binomial(
      @NonNegative @LTLengthOf("BinomialTest.factorials") int n,
      @NonNegative @LessThan("#1 + 1") int k) {
        float __cfwr_entry78 = -5.24f;

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
      protected boolean __cfwr_handle147() {
        return null;
        for (int __cfwr_i15 = 0; __cfwr_i15 < 2; __cfwr_i15++) {
            return null;
        }
        Boolean __cfwr_node36 = null;
        return true;
    }
    private static Character __cfwr_handle957(char __cfwr_p0, boolean __cfwr_p1, char __cfwr_p2) {
        Character __cfwr_entry55 = null;
        return null;
    }
    protected static boolean __cfwr_compute235(char __cfwr_p0) {
        while (true) {
            for (int __cfwr_i26 = 0; __cfwr_i26 < 7; __cfwr_i26++) {
            while (false) {
            while ((-20.50 - -67.09f)) {
            for (int __cfwr_i71 = 0; __cfwr_i71 < 10; __cfwr_i71++) {
            if ((-49.86 / (-379L & 841L)) && (true | -678L)) {
            while (true) {
            for (int __cfwr_i69 = 0; __cfwr_i69 < 8; __cfwr_i69++) {
            try {
            while (false) {
            try {
            while (false) {
            short __cfwr_entry71 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e35) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e7) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        return 50.31;
        return ((8.74f + null) | ('6' >> 'R'));
    }
}
