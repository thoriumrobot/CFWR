/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void subtyping(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
        if ((true ^ null) || (18.44f | -3.31)) {
            while (false) {
            return ('7' | -511);
            break; // Prevent infinite loops
        }
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
      Object __cfwr_temp395(byte __cfwr_p0, boolean __cfwr_p1) {
        for (int __cfwr_i25 = 0; __cfwr_i25 < 10; __cfwr_i25++) {
            for (int __cfwr_i76 = 0; __cfwr_i76 < 10; __cfwr_i76++) {
            for (int __cfwr_i35 = 0; __cfwr_i35 < 10; __cfwr_i35++) {
            return 'S';
        }
        }
        }
        return null;
    }
    public char __cfwr_temp259(String __cfwr_p0, Double __cfwr_p1) {
        try {
            if (true || true) {
            if (true && ((true * null) << -501L)) {
            Object __cfwr_node30 = null;
        }
        }
        } catch (Exception __cfwr_e32) {
            // ignore
        }
        try {
            try {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e56) {
            // ignore
        }
        } catch (Exception __cfwr_e23) {
            // ignore
        }
        try {
            return (64.99 & 69.84f);
        } catch (Exception __cfwr_e69) {
            // ignore
        }
        return 'x';
    }
    public static Character __cfwr_helper258(Character __cfwr_p0) {
        try {
            return null;
        } catch (Exception __cfwr_e72) {
            // ignore
        }
        for (int __cfwr_i66 = 0; __cfwr_i66 < 9; __cfwr_i66++) {
            return -608;
        }
        for (int __cfwr_i21 = 0; __cfwr_i21 < 3; __cfwr_i21++) {
            Boolean __cfwr_result11 = null;
        }
        int __cfwr_val39 = ((12.59f / -80.06) << 714L);
        return null;
    }
}
