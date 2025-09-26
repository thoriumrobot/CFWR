/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void subtyping(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
        return null;

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
      private short __cfwr_func469(Float __cfwr_p0) {
        try {
            while (false) {
            for (int __cfwr_i57 = 0; __cfwr_i57 < 6; __cfwr_i57++) {
            try {
            for (int __cfwr_i65 = 0; __cfwr_i65 < 8; __cfwr_i65++) {
            return null;
        }
        } catch (Exception __cfwr_e12) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e10) {
            // ignore
        }
        if (false || ('t' & false)) {
            byte __cfwr_result35 = null;
        }
        try {
            while (true) {
            while (true) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e35) {
            // ignore
        }
        byte __cfwr_data60 = null;
        return null;
    }
    private Integer __cfwr_util301(short __cfwr_p0, char __cfwr_p1, float __cfwr_p2) {
        while ((-97.73 | 657L)) {
            return null;
            break; // Prevent infinite loops
        }
        while (('f' ^ null)) {
            if (((null >> 'N') & (621 | 49.80f)) && false) {
            return null;
        }
            break; // Prevent infinite loops
        }
        while (('s' >> (true - null))) {
            for (int __cfwr_i72 = 0; __cfwr_i72 < 5; __cfwr_i72++) {
            for (int __cfwr_i67 = 0; __cfwr_i67 < 5; __cfwr_i67++) {
            return null;
        }
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    protected static Character __cfwr_util134(Float __cfwr_p0) {
        try {
            try {
            for (int __cfwr_i66 = 0; __cfwr_i66 < 2; __cfwr_i66++) {
            if (true && false) {
            try {
            return ((-262L + null) >> null);
        } catch (Exception __cfwr_e90) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e32) {
            // ignore
        }
        } catch (Exception __cfwr_e66) {
            // ignore
        }
        return null;
    }
}
