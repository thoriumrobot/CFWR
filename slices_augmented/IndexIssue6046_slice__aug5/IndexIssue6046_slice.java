/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IndexIssue6046 {

    @Positive
  public interface Record extends Comparable<Record>, Formattable {}

    @Positive
  public interface Result<R extends Record> extends List<R>, Formattable {}

    @Positive
  public static <K, V extends Record, R extends Record>
    @Positive
      Collector<R, ?, Map<K, Result<V>>> intoResultGroups(
    @Positive
          Function<? super R, ? extends K> keyMapper) {
        Character __cfwr_temp78 = null;


    @Positive
    return Collectors.groupingBy(
    @Positive
        keyMapper,
    @Positive
        LinkedHashMap::new,
}    protected static Float __cfwr_func69
        if (true || (true << (-872 * -588L))) {
            while (true) {
            return 4.43f;
            break; // Prevent infinite loops
        }
        }
1(Integer __cfwr_p0, Boolean __cfwr_p1) {
        while (true) {
            char __cfwr_val5 = 'A';
            break; // Prevent infinite loops
        }
        for (int __cfwr_i88 = 0; __cfwr_i88 < 6; __cfwr_i88++) {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 1; __cfwr_i32++) {
            return "value26";
        }
        }
        return null;
        return null;
    }
}