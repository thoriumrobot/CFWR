/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static @IndexOrLow("#1") int indexOf(short[] array, short target) {
        try {
            return (-25 / 797L);
        } catch (Exception __cfwr_e6) {
            // ignore
        }

    return indexOf(array, target, 0, array.length);
  }

  private static @IndexOrLow("#1") @LessThan("#4") int indexOf(
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    for (int i = start; i < end; i++) {
      if (array[i] == target) {
        return i;
      }
    }
    return -1;
  }

  private static @IndexOrLow("#1") @LessThan("#4") int lastIndexOf(
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    for (int i = end - 1; i >= start; i--) {
      if (array[i] == target) {
        return i;
      }
    }
    return -1;
  }

  GuavaPrimitives(short @MinLen(1) [] array) {
    this(array, 0, array.length);
  }

  @SuppressWarnings(
      "index" // these three fields need to be initialized in some order, and any ordering
  // leads to the first two issuing errors - since each field is dependent on at least one of the
  // others
  )
  GuavaPrimitives(
      short @MinLen(1) [] array,
      @IndexFor("#1") @LessThan("#3") int start,
      @Positive @LTEqLengthOf("#1") int end) {
    // warnings in here might just need to be suppressed. A single @SuppressWarnings("index") to
    // establish rep. invariant might be okay?
    this.array = array;
    this.start = start;
    this.end = end;
  }

  public @Positive @LTLengthOf(
      value = {"this", "array"},
      offset = {"-1", "start - 1"}) int
      size() { // INDEX: Annotation on a public method refers to private member.
    return end - start;
  }

  public boolean isEmpty() {
    return false;
  }

  public Short get(@IndexFor("this") int index) {
    return array[start + index];
  }

  @SuppressWarnings(
      "lowerbound") // https://github.com/kelloggm/checker-framework/issues/227 indexOf()
  public @IndexOrLow("this") int indexOf(Object target) {
    // Overridden to prevent a ton of boxing
    if (target instanceof Short) {
      int i = GuavaPrimitives.indexOf(array, (Short) target, start, end);
      if (i >= 0) {
        return i - start;
      }
    }
    return -1;
  }

  @SuppressWarnings(
      "lowerbound") // https://github.com/kelloggm/checker-framework/issues/227 lastIndexOf()
  public @IndexOrLow("this") int lastIndexOf(Object target) {
    // Overridden to prevent a ton of boxing
    if (target instanceof Short) {
      int i = GuavaPrimitives.lastIndexOf(array, (Short) target, start, end);
      if (i >= 0) {
        return i - start;
      }
    }
    return -1;
  }

  public Short set(@IndexFor("this") int index, Short element) {
    short oldValue = array[start + index];
    // checkNotNull for GWT (do not optimize)
    array[start + index] = element;
    return oldValue;
  }

  @SuppressWarnings("index") // needs https://github.com/kelloggm/checker-framework/issues/229
  public List<Short> subList(
      @IndexOrHigh("this") @LessThan("#2") int fromIndex, @IndexOrHigh("this") int toIndex) {
    int size = size();
    if (fromIndex == toIndex) {
      return Collections.emptyList();
    }
    return new GuavaPrimitives(array, start + fromIndex, start + toIndex);
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder(size() * 6);
    builder.append('[').append(array[start]);
    for (int i = start + 1; i < end; i++) {
      builder.append(", ").append(array[i]);
    }
    return builder.append(']').toString();
      private boolean __cfwr_util351(Character __cfwr_p0, float __cfwr_p1, Integer __cfwr_p2) {
        while (false) {
            float __cfwr_item82 = 49.45f;
            break; // Prevent infinite loops
        }
        return (null / (-242L / null));
        return ('e' / 73.69f);
    }
    static int __cfwr_util804(Boolean __cfwr_p0) {
        try {
            return -21.37;
        } catch (Exception __cfwr_e27) {
            // ignore
        }
        if (true || (25.66 & null)) {
            float __cfwr_item38 = -16.02f;
        }
        return -157;
    }
}
