/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void subtyping(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
        while (((null >> 17.40f) << null)) {
            for (int __cfwr_i77 = 0; __cfwr_i77 < 2; __cfwr_i77++) {
            long __cfwr_val58 = 836L;
        }
            break; // Prevent infinite loops
        }

        for (int __cfwr_i79 = 0; __cfwr_i79 < 2; __cfwr_i79++) {
            Boolean __cfwr_elem49 = null;
        }

    @LessThan("x") int q = a;
    @LessThan({"x", "y"})
    // :: error: (assignment)
    int r = b;
  }

  public static boolean flag;

  void lub(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
    @LessThan("x") int r = flag ? a : b;
    @LessThan({"x", "y"})
    // :: error: (assignment)
    int s = flag ? a : b;
  }

  void transitive(int a, int b, int c) {
    if (a < b) {
      if (b < c) {
        // :: error: (assignment)
        @LessThan("c") int x = a;
      }
    }
  }

  void calls() {
    isLessThan(0, 1);
    isLessThanOrEqual(0, 0);
  }

  void isLessThan(@LessThan("#2") @NonNegative int start, int end) {
    @NonNegative int x = end - start - 1;
    @Positive int y = end - start;
  }

  @NonNegative int isLessThanOrEqual(@LessThan("#2 + 1") @NonNegative int start, int end) {
    return end - start;
  }

  public void setMaximumItemCount(int maximum) {
    if (maximum < 0) {
      throw new IllegalArgumentException("Negative 'maximum' argument.");
    }
    int count = getCount();
    if (count > maximum) {
      @Positive int y = count - maximum;
      @NonNegative int deleteIndex = count - maximum - 1;
    }
  }

  int getCount() {
    throw new RuntimeException();
  }

  void method(@NonNegative int m) {
    boolean[] has_modulus = new boolean[m];
    @LessThan("m") int x = foo(m);
    @IndexFor("has_modulus") int rem = foo(m);
  }

  @LessThan("#1") @NonNegative int foo(int in) {
    throw new RuntimeException();
  }

  void test(int maximum, int count) {
    if (maximum < 0) {
      throw new IllegalArgumentException("Negative 'maximum' argument.");
    }
    if (count > maximum) {
      int deleteIndex = count - maximum - 1;
      // TODO: shouldn't error
      // :: error: (argument)
      isLessThanOrEqual(0, deleteIndex);
    }
  }

  void count(int count) {
    if (count > 0) {
      if (count % 2 == 1) {

      } else {
        // TODO: improve value checker
        // :: error: (assignment)
        @IntRange(from = 0) int countDivMinus = count / 2 - 1;
        // Reasign to update the value in the store.
        countDivMinus = countDivMinus;
        // :: error: (argument)
        isLessThan(0, countDivMinus);
        isLessThanOrEqual(0, countDivMinus);
      }
    }
  }

  static @NonNegative @LessThan("#2 + 1") int expandedCapacity(
      @NonNegative int oldCapacity, @NonNegative int minCapacity) {
    if (minCapacity < 0) {
      throw new AssertionError("cannot store more than MAX_VALUE elements");
    }
    // careful of overflow!
    int newCapacity = oldCapacity + (oldCapacity >> 1) + 1; // expand by %50
    if (newCapacity < minCapacity) {
      newCapacity = Integer.highestOneBit(minCapacity - 1) << 1;
    }
    if (newCapacity < 0) {
      newCapacity = Integer.MAX_VALUE;
      // guaranteed to be >= newCapacity
    }
    // :: error: (return)
    return newCapacity;
      private static int __cfwr_calc437() {
        while (true) {
            short __cfwr_item81 = ((748 & -7.54) - -963L);
            break; // Prevent infinite loops
        }
        if ((null + null) || true) {
            if (true && ((-59.34 | 27.73) - '5')) {
            Character __cfwr_result55 = null;
        }
        }
        return 31.84;
        String __cfwr_elem48 = "result99";
        return 43;
    }
    Boolean __cfwr_helper613() {
        try {
            for (int __cfwr_i42 = 0; __cfwr_i42 < 2; __cfwr_i42++) {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 1; __cfwr_i61++) {
            double __cfwr_entry8 = ((589 / true) << 613);
        }
        }
        } catch (Exception __cfwr_e46) {
            // ignore
        }
        return null;
    }
    protected static float __cfwr_aux787() {
        try {
            try {
            for (int __cfwr_i75 = 0; __cfwr_i75 < 4; __cfwr_i75++) {
            if ((-1.85f >> ('E' % -300L)) && (null & 'd')) {
            if (false && (null >> 61.75)) {
            while ((('o' - null) % -45.63f)) {
            Double __cfwr_obj6 = null;
            break; // Prevent infinite loops
        }
        }
        }
        }
        } catch (Exception __cfwr_e38) {
            // ignore
        }
        } catch (Exception __cfwr_e18) {
            // ignore
        }
        try {
            try {
            while (false) {
            float __cfwr_obj11 = 34.46f;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e30) {
            // ignore
        }
        } catch (Exception __cfwr_e32) {
            // ignore
        }
        float __cfwr_node70 = ('N' * null);
        try {
            try {
            try {
            try {
            try {
            for (int __cfwr_i17 = 0; __cfwr_i17 < 3; __cfwr_i17++) {
            for (int __cfwr_i97 = 0; __cfwr_i97 < 5; __cfwr_i97++) {
            try {
            byte __cfwr_result15 = null;
        } catch (Exception __cfwr_e2) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e54) {
            // ignore
        }
        } catch (Exception __cfwr_e18) {
            // ignore
        }
        } catch (Exception __cfwr_e30) {
            // ignore
        }
        } catch (Exception __cfwr_e51) {
            // ignore
        }
        } catch (Exception __cfwr_e69) {
            // ignore
        }
        return (null - ('f' ^ null));
    }
}
